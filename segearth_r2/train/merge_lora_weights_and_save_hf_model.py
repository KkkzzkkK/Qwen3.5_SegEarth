import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import glob
import copy
import json
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor

from segearth_r2.model import *
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--model_path", default="./save_model/SegEarth-R2"
    )

    parser.add_argument(
        "--vision_tower_mask", default="./pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    )
    parser.add_argument(
        "--mask_config", default="./segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )

    parser.add_argument("--lora_enable", default=True, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)
    
    return parser.parse_args(args)


def find_linear_layers(model, lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj', 'gate_proj', 'up_proj', 'down_proj'], train_module_list=[]):
    cur_train_module_list = copy.deepcopy(train_module_list)
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask", "visual"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls)
            and all(
                        [
                            x not in name
                            for x in cur_train_module_list
                        ]
                    )
                    and any([x in name for x in lora_target_modules])):

            lora_module_names.add(name)
            
    return sorted(list(lora_module_names))

def load_pretrained_model(model_path, model_args, mask_config='/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    has_adapter_checkpoint = os.path.exists(adapter_config_path)

    base_model_path = model_path
    if has_adapter_checkpoint:
        with open(adapter_config_path, "r", encoding="utf-8") as fp:
            adapter_config = json.load(fp)
        base_model_path = adapter_config.get("base_model_name_or_path", model_path)

    tokenizer_source_path = model_path if has_adapter_checkpoint else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_path, use_fast=True)
    tokenizer.add_tokens("[SEG]")
    model = SegEarthR2.from_pretrained(base_model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)

    model.initialize_vision_modules(model_args)

    vision_tower_mask = model.get_vision_tower_mask()
    vision_tower_mask.to(device=device)

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
        "seg_scale_projectors", "seg_layer_weights",
    ]

    model.resize_token_embeddings(len(tokenizer))

    if has_adapter_checkpoint:
        # Load non-LoRA trainable params BEFORE PEFT wrapping so that key
        # names match the base model directly (no "base_model.model." prefix).
        non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
        if os.path.exists(non_lora_path):
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            cleaned = {}
            for k, v in non_lora_state.items():
                for prefix in ["base_model.model.", "model."]:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                cleaned[k] = v
            msg = model.load_state_dict(cleaned, strict=False)
            print(f"Loaded non-LoRA trainables into base model: {len(cleaned)} keys "
                  f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
            if msg.unexpected_keys:
                print(f"  unexpected keys: {msg.unexpected_keys[:10]}...")
        else:
            print(f"WARNING: {non_lora_path} not found. "
                f"Non-LoRA trainable params (pixel_decoder, predictor, "
                f"SEG_token_projector, seg_scale_projectors, seg_layer_weights) "
                f"may not be restored from training.")

        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
        model = model.merge_and_unload()
    elif model_args.lora_enable:
        lora_r = model_args.lora_r
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(model, model_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    return tokenizer, model

def main(args):
    args = parse_args(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    for k, v in model.state_dict().items():
        print(k)
        state_dict[k] = v
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)

    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    processor_source_path = args.model_path
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as fp:
            adapter_config = json.load(fp)
        processor_source_path = adapter_config.get("base_model_name_or_path", args.model_path)

    try:
        processor = AutoProcessor.from_pretrained(processor_source_path)
        processor.save_pretrained(args.save_path)
    except Exception:
        for cfg_name in ("preprocessor_config.json", "processor_config.json",
                         "chat_template.json"):
            src = os.path.join(processor_source_path, cfg_name)
            dst = os.path.join(args.save_path, cfg_name)
            if os.path.exists(src):
                shutil.copy(src, dst)

    tokenizer.save_pretrained(args.save_path)

    check_tok = AutoTokenizer.from_pretrained(args.save_path, use_fast=True)
    if check_tok.convert_tokens_to_ids("[SEG]") is None:
        raise RuntimeError(
            "[SEG] token is missing after export. "
            "Please check tokenizer/processor save order and source tokenizer files."
        )
    
if __name__ == "__main__":
    main(sys.argv[1:])
