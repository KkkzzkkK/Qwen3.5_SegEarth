import os
import json
import torch
import shutil
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.distributed as dist
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available, is_accelerate_available
try:
    from transformers.utils import is_torch_tpu_available
except ImportError:
    from transformers.utils import is_torch_xla_available

    def is_torch_tpu_available(check_device=False):
        try:
            return is_torch_xla_available(check_device=check_device)
        except TypeError:
            return is_torch_xla_available()
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from fvcore.nn import FlopCountAnalysis, parameter_count
from deepspeed.profiling.flops_profiler import get_model_profile

import torch

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    logging,
    strtobool,
)


def is_safetensors_available():
    try:
        import safetensors  # noqa: F401
        return True
    except Exception:
        return False


def is_torch_compile_available():
    return hasattr(torch, "compile")


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            save_checkpoint_signature = inspect.signature(super(LLaVATrainer, self)._save_checkpoint)
            if "metrics" in save_checkpoint_signature.parameters:
                super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            else:
                super(LLaVATrainer, self)._save_checkpoint(model, trial)

            # Save non-LoRA trainable params alongside checkpoint so the
            # merge script can restore pixel_decoder / predictor /
            # SEG_token_projector / lm_head.
            if hasattr(self.model, "peft_config"):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                ckpt_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                ckpt_dir = os.path.join(run_dir, ckpt_folder)
                non_lora_state = {
                    k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
                    for k, v in self.model.named_parameters()
                    if v.requires_grad and "lora_" not in k
                }
                if non_lora_state and self.args.local_rank in (0, -1):
                    torch.save(
                        non_lora_state,
                        os.path.join(ckpt_dir, "non_lora_trainables.bin"),
                    )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def create_optimizer(self):
        """Split parameters into two LR groups: seg modules (high LR) vs rest (base LR)."""
        from torch.optim import AdamW

        seg_module_keywords = [
            "SEG_token_projector", "pixel_decoder", "predictor",
            "seg_scale_projectors",
            "seg_layer_weights",
        ]
        seg_learning_rate = getattr(self.args, 'seg_learning_rate', 1e-4)

        seg_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in seg_module_keywords):
                seg_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = []
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            })
        if seg_params:
            optimizer_grouped_parameters.append({
                "params": seg_params,
                "lr": seg_learning_rate,
                "weight_decay": self.args.weight_decay,
            })

        print(f"[Grouped LR] LoRA/other params: lr={self.args.learning_rate}, "
              f"count={len(other_params)}")
        print(f"[Grouped LR] Seg module params: lr={seg_learning_rate}, "
              f"count={len(seg_params)}")

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()

    def _get_bad_sample_log_path(self):
        custom_path = getattr(self.args, 'bad_sample_log_file', None)
        if custom_path:
            return custom_path
        return os.path.join(self.args.output_dir, 'bad_samples.jsonl')

    def _record_bad_training_samples(self, outputs):
        if not getattr(self.args, 'record_bad_samples', False):
            return
        if not self.is_world_process_zero():
            return

        if isinstance(outputs, dict):
            sample_diagnostics = outputs.get('sample_diagnostics')
            batch_loss_dice = outputs.get('loss_dice')
        else:
            sample_diagnostics = getattr(outputs, 'sample_diagnostics', None)
            batch_loss_dice = getattr(outputs, 'loss_dice', None)

        if not sample_diagnostics:
            return

        threshold = float(getattr(self.args, 'bad_sample_dice_threshold', 0.0) or 0.0)
        bad_samples = [
            sample for sample in sample_diagnostics
            if float(sample.get('loss_dice_weighted', 0.0)) >= threshold
        ]
        if not bad_samples:
            return

        log_path = self._get_bad_sample_log_path()
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        epoch = float(self.state.epoch) if self.state.epoch is not None else None
        batch_loss_dice_value = None
        if torch.is_tensor(batch_loss_dice):
            batch_loss_dice_value = float(batch_loss_dice.item())
        elif batch_loss_dice is not None:
            batch_loss_dice_value = float(batch_loss_dice)

        with open(log_path, 'a', encoding='utf-8') as f:
            for sample in bad_samples:
                record = dict(sample)
                record['global_step'] = int(self.state.global_step)
                record['epoch'] = epoch
                record['threshold'] = threshold
                record['batch_loss_dice'] = batch_loss_dice_value
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        global_step = self.state.global_step
        inputs['global_step'] = global_step
        
        outputs = model(**inputs)

        past_index = getattr(self.args, "past_index", -1)
        if past_index >= 0:
            self._past = outputs[past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'loss_dice' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                self._record_bad_training_samples(outputs)
                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss