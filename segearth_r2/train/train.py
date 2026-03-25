import os
import sys
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from transformers import AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
import warnings
import copy
import json
from deepspeed.profiling.flops_profiler import get_model_profile

from segearth_r2.datasets.dataset import *
from llava_trainer import LLaVATrainer
from segearth_r2.model.language_model.llava_phi import SegEarthR2

warnings.filterwarnings('ignore')
local_rank = None


def maybe_enable_legacy_torch_resume_compat():
    flag = os.environ.get("ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME", "").strip().lower()
    if flag not in {"1", "true", "yes", "y", "on"}:
        return

    def _noop_check_torch_load_is_safe():
        return None

    def _skip_load_rng_state(self, checkpoint):
        return None

    try:
        import transformers.trainer as hf_trainer_module
        hf_trainer_module.check_torch_load_is_safe = _noop_check_torch_load_is_safe
        hf_trainer_module.Trainer._load_rng_state = _skip_load_rng_state
    except Exception:
        pass

    try:
        from transformers.utils import import_utils as hf_import_utils
        hf_import_utils.check_torch_load_is_safe = _noop_check_torch_load_is_safe
    except Exception:
        pass

    print("[WARN] ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME is enabled. "
          "Bypassing transformers torch.load safety guard for checkpoint resume.")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/root/autodl-tmp/qwen")

    version: Optional[str] = field(default="qwen3_5")

    freeze_backbone: bool = field(default=False)
    train_clip_backbone: bool = field(default=False)
    train_swin_backbone: bool = field(default=False)
    swin_trainable_stages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated Swin stage indices to train (0-3), e.g. '2,3'."}
    )

    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    with_norm: bool = field(default=True)
    with_layernorm: bool = field(default=False)
    skip_init_vision: bool = field(default=False)
    swin_type: Optional[str] = field(default="base")
    projector_outdim: Optional[int] = field(default=2048)
    mm_projector_type: Optional[str] = field(default="swin_conv")
    model_version: Optional[str] = field(default="v1")
    load_mask2former: bool = field(default=True)
    mask_config: Optional[str] = field(default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    mm_use_im_patch_token: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=False)

@dataclass
class DataArguments:
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    base_data_path: str = '/data1/xzp/data'
    data_ratio: str = '1'  
    switch_bs: int = 4 # 16
    fix_dataset_len: int = 0
    segmentation: bool = True
    require_seg_in_each_sample: bool = True
    drop_invalid_seg_sample: bool = True
    compact_answer_supervision: bool = True
    answer_in_user_prompt: bool = True
    max_seg_per_sample: int = 0

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    dataloader_prefetch_factor: int = field(default=2)
    dataloader_num_workers: int = field(default=4)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=False)
    deepspeed: Optional[str] = field(default=None)
    use_attention_loss: bool = field(default=False)
    attn_implementation: Optional[str] = field(default='auto')
    adam_epsilon: float = field(default=1e-6)
    
    output_dir: Optional[str] = field(default="output/model")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    dataloader_drop_last: bool = True
    llm_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    seg_learning_rate: float = 1e-4
    record_bad_samples: bool = True
    bad_sample_dice_threshold: float = 40.0
    bad_sample_log_file: Optional[str] = None


def resolve_attn_implementation(requested_impl, use_attention_loss=False):
    if requested_impl and requested_impl != 'auto':
        return requested_impl
    if use_attention_loss:
        return 'eager'
    if importlib.util.find_spec('flash_attn') is not None:
        return 'flash_attention_2'
    return 'sdpa'


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

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
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def set_swin_trainable_stages(vision_tower_mask, stage_spec: Optional[str]):
    if not stage_spec:
        vision_tower_mask.requires_grad_(True)
        print("Swin backbone train mode: full unfreeze")
        return

    stage_tokens = [x.strip() for x in stage_spec.split(',') if x.strip()]
    trainable_stages = []
    for token in stage_tokens:
        stage_id = int(token)
        if stage_id < 0 or stage_id > 3:
            raise ValueError(f"Invalid swin_trainable_stages value: {stage_id}. Expected stage index in [0, 3].")
        trainable_stages.append(stage_id)

    trainable_stages = sorted(set(trainable_stages))
    if not trainable_stages:
        raise ValueError("swin_trainable_stages is set but no valid stage index was provided.")

    vision_tower_mask.requires_grad_(False)
    for name, param in vision_tower_mask.named_parameters():
        if any(f"layers.{stage_id}." in name for stage_id in trainable_stages):
            param.requires_grad = True
        elif name.startswith("norm") or ".norm." in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in vision_tower_mask.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_tower_mask.parameters())
    print(
        f"Swin backbone train mode: partial unfreeze stages={trainable_stages}; "
        f"trainable params={trainable}/{total}"
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)

        # PEFT's save_pretrained only saves adapter weights.
        # Manually export non-LoRA trainable params (pixel_decoder,
        # predictor, SEG_token_projector, lm_head, etc.) so the
        # merge script can restore them.
        if hasattr(trainer.model, "peft_config"):
            non_lora_state = {
                k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
                for k, v in trainer.model.named_parameters()
                if v.requires_grad and "lora_" not in k
            }
            if non_lora_state and trainer.args.local_rank in (0, -1):
                torch.save(
                    non_lora_state,
                    os.path.join(output_dir, "non_lora_trainables.bin"),
                )
                print(f"Saved {len(non_lora_state)} non-LoRA trainable params "
                      f"to {output_dir}/non_lora_trainables.bin")
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def make_unify_datamodule(clip_image_processor, tokenizer, data_args, training_args, qwen_vision_cfg=None):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split('||')
    data_ratio = [int(data_) for data_ in data_ratio]
    datasets = []
    if data_ratio[0] != 0:
        LaSeRSTrainDataset = LaSeRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args)
        datasets += [LaSeRSTrainDataset] * data_ratio[0]
    
    print(f'the dataset ratio is: {data_ratio}')
    train_dataset = UnifyDatasetSingleDatasetForBatch(datasets, data_ratio, data_args.switch_bs, fix_dataset_len=data_args.fix_dataset_len)
    print(f'total unify datasest number is {len(train_dataset)}')

    collator_kwargs = dict(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    if qwen_vision_cfg is not None:
        collator_kwargs.update(qwen_vision_cfg)
    data_collator = DataCollatorForCOCODatasetV2(**collator_kwargs)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)) # 用不着？

    mask_cfg = get_mask_config(config=model_args.mask_config)
    bnb_model_from_pretrained_args = {}

    attn_implementation = resolve_attn_implementation(
        training_args.attn_implementation,
        use_attention_loss=training_args.use_attention_loss,
    )

    load_path = model_args.model_name_or_path
    adapter_config_path = os.path.join(load_path, "adapter_config.json")
    has_adapter_checkpoint = os.path.exists(adapter_config_path)
    base_model_path = load_path
    if has_adapter_checkpoint:
        with open(adapter_config_path, "r", encoding="utf-8") as fp:
            adapter_config = json.load(fp)
        base_model_path = adapter_config.get("base_model_name_or_path", load_path)

    model = SegEarthR2.from_pretrained(
        base_model_path,
        mask_decoder_cfg=mask_cfg,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        **bnb_model_from_pretrained_args
                )
    model.config.use_attention_loss = training_args.use_attention_loss
    print(
        f"Using attention implementation: {attn_implementation}; "
        f"attention loss enabled: {training_args.use_attention_loss}"
    )

    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower_mask if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt, model_args)

    model.llm_loss_weight = training_args.llm_loss_weight
    model.mask_loss_weight = training_args.mask_loss_weight

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer_source_path = load_path if has_adapter_checkpoint else base_model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        fallback_key = None
        for key in ["qwen3_5", "vicuna_v1", "default", "v0"]:
            if key in conversation_lib.conv_templates:
                fallback_key = key
                break
        if fallback_key is None:
            fallback_key = next(iter(conversation_lib.conv_templates))
        conversation_lib.default_conversation = conversation_lib.conv_templates[fallback_key]

    # Initialize Swin for segmentation path (Qwen3.5 ViT already loaded)
    if model_args.vision_tower_mask is not None:
        model.initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower_mask = model.get_vision_tower_mask()
        vision_tower_mask.to(
            dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32),
            device=training_args.device
        )
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        # Freeze Qwen3.5 visual encoder by default
        if not model_args.train_clip_backbone:
            model.model.visual.requires_grad_(False)
        # Freeze Swin by default
        if not model_args.train_swin_backbone:
            model.get_vision_tower_mask().requires_grad_(False)
        elif model_args.swin_trainable_stages:
            set_swin_trainable_stages(model.get_vision_tower_mask(), model_args.swin_trainable_stages)

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
        "seg_scale_projectors", "seg_layer_weights",
    ]

    if model_args.train_swin_backbone:
        train_module_list.append('vision_tower_mask')

    if training_args.lora_enable:
        lora_r = training_args.lora_r
        lora_alpha = training_args.lora_alpha
        lora_dropout = training_args.lora_dropout
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
        if has_adapter_checkpoint:
            non_lora_path = os.path.join(load_path, "non_lora_trainables.bin")
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
                print(
                    f"Loaded non-LoRA trainables from {non_lora_path}: {len(cleaned)} keys "
                    f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
                )
            else:
                print(f"WARNING: non-LoRA trainables not found at {non_lora_path}")

            model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
            print(f"Loaded LoRA adapter weights from {load_path} for continued training")
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in train_module_list
                ]):

                p.requires_grad = True

    model.get_special_token(SEG=tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'], EOS=tokenizer.eos_token_id)

    # Use Qwen3.5's processor for image preprocessing
    processor_source_path = base_model_path
    qwen_processor = AutoProcessor.from_pretrained(processor_source_path)
    # Keep multimodal token budget under model_max_length to avoid mRoPE shape mismatches.
    # Qwen vision uses 28x28 pixel area per visual token.
    # NOTE: model_max_length can be large for text, but vision tokens should stay capped
    # to keep memory predictable during training.
    env_budget = os.environ.get("QWEN_IMAGE_TOKEN_BUDGET", "").strip()
    if env_budget:
        image_token_budget = max(64, int(env_budget))
    else:
        image_token_budget = max(256, int(training_args.model_max_length * 0.6))
        image_token_budget = min(image_token_budget, 1536)

    try:
        qwen_processor.image_processor.max_pixels = image_token_budget * 28 * 28
        print(
            f"Qwen image processor max_pixels set to {qwen_processor.image_processor.max_pixels} "
            f"(vision token budget={image_token_budget})"
        )
    except Exception:
        pass

    qwen_vision_cfg = dict(
        image_token_id=model.config.image_token_id,
        vision_start_token_id=model.config.vision_start_token_id,
        vision_end_token_id=model.config.vision_end_token_id,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
    )

    data_module = make_unify_datamodule(
        clip_image_processor=qwen_processor.image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        qwen_vision_cfg=qwen_vision_cfg,
    )
    training_args.dataloader_drop_last = True
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        **data_module,
    )
    try:
        trainer = LLaVATrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = LLaVATrainer(processing_class=tokenizer, **trainer_kwargs)

    maybe_enable_legacy_torch_resume_compat()

    explicit_resume = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    if explicit_resume:
        trainer.train(resume_from_checkpoint=explicit_resume)
    else:
        resume_arg = getattr(training_args, "resume_from_checkpoint", None)
        if resume_arg:
            trainer.train(resume_from_checkpoint=resume_arg)
        else:
            trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()