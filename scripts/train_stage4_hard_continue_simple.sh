#!/usr/bin/env bash
set -euo pipefail

# Simple hard-sample continue script from stage4 checkpoint.
# Keep only explicit, user-controlled knobs: load/save path, rounds, steps, lr.

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3.10)
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python)
    else
        echo "No Python interpreter found. Set PYTHON_BIN or install python3."
        exit 1
    fi
fi

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_PORT=${MASTER_PORT:-29500}

# -------- Required paths --------
# stage4 checkpoint dir, e.g. /root/autodl-tmp/output4/segearth_r2_lora/checkpoint-XXXX
LOAD_CKPT_PATH=${LOAD_CKPT_PATH:-/root/autodl-tmp/output4/segearth_r2_lora_hard_data_mix_stage3_out_lora_aug_data_hard_1to1_from_stage3_out/checkpoint-8100}
# hard-mining dataset root with train/annotations/train_data.json
HARD_DATA_PATH=${HARD_DATA_PATH:-/root/autodl-tmp/data}
# new output dir
SAVE_OUTPUT_DIR=${SAVE_OUTPUT_DIR:-/root/autodl-tmp/output4/segearth_r2_lora_hard_data_mix_stage3_out_lora_aug_data_hard_1to1_from_stage3_out}

if [[ -z "${LOAD_CKPT_PATH}" ]]; then
    echo "[ERROR] LOAD_CKPT_PATH is required."
    exit 1
fi
if [[ ! -d "${LOAD_CKPT_PATH}" ]]; then
    echo "[ERROR] LOAD_CKPT_PATH does not exist: ${LOAD_CKPT_PATH}"
    exit 1
fi
if [[ ! -f "${HARD_DATA_PATH}/train/annotations/train_data.json" ]]; then
    echo "[ERROR] Hard dataset not found: ${HARD_DATA_PATH}/train/annotations/train_data.json"
    exit 1
fi

# -------- Training control --------
ROUNDS=${ROUNDS:-2}
ROUND_STEPS=${ROUND_STEPS:-2500}
EXTRA_STEPS=${EXTRA_STEPS:-$((ROUNDS * ROUND_STEPS))}

# -------- Runtime --------
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero1.json}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-3072}
QWEN_IMAGE_TOKEN_BUDGET=${QWEN_IMAGE_TOKEN_BUDGET:-1024}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
MAX_SEG_PER_SAMPLE=${MAX_SEG_PER_SAMPLE:-10}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS:-False}
TRAIN_SWIN_BACKBONE=${TRAIN_SWIN_BACKBONE:-True}
SWIN_TRAINABLE_STAGES=${SWIN_TRAINABLE_STAGES:-0,1,2,3}
RECORD_BAD_SAMPLES=${RECORD_BAD_SAMPLES:-True}
BAD_SAMPLE_DICE_THRESHOLD=${BAD_SAMPLE_DICE_THRESHOLD:-25}

# -------- Optim --------
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SEG_LEARNING_RATE=${SEG_LEARNING_RATE:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-100}

CURRENT_STEP=$(
    "${PYTHON_BIN}" - <<'PY' "${LOAD_CKPT_PATH}"
import json
import os
import re
import sys
ckpt = sys.argv[1]
state_path = os.path.join(ckpt, "trainer_state.json")
step = None
if os.path.exists(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            step = int(json.load(f).get("global_step", 0))
    except Exception:
        step = None
if step is None or step <= 0:
    m = re.search(r"checkpoint-(\\d+)$", ckpt)
    step = int(m.group(1)) if m else 0
print(step)
PY
)
TARGET_MAX_STEPS=$((CURRENT_STEP + EXTRA_STEPS))

# Model-only resume to avoid optimizer-state mismatch when trainable set changes.
MODEL_NAME_OR_PATH="${LOAD_CKPT_PATH}"
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
if [[ "${RESUME_MODEL_ONLY}" != "1" && "${RESUME_MODEL_ONLY}" != "true" ]]; then
    export RESUME_FROM_CHECKPOINT="${LOAD_CKPT_PATH}"
else
    unset RESUME_FROM_CHECKPOINT || true
fi

LORA_R=${LORA_R:-}
if [[ -z "${LORA_R}" ]]; then
    ADAPTER_CONFIG_PATH="${MODEL_NAME_OR_PATH}/adapter_config.json"
    if [[ -f "${ADAPTER_CONFIG_PATH}" ]]; then
        LORA_R=$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("r", 16))' "${ADAPTER_CONFIG_PATH}")
    else
        LORA_R=16
    fi
fi

export QWEN_IMAGE_TOKEN_BUDGET
export ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME=${ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME:-1}

echo "[Stage4-Hard-Continue] load_ckpt: ${LOAD_CKPT_PATH}"
echo "[Stage4-Hard-Continue] save_output: ${SAVE_OUTPUT_DIR}"
echo "[Stage4-Hard-Continue] hard_data: ${HARD_DATA_PATH}"
echo "[Stage4-Hard-Continue] max_steps: ${TARGET_MAX_STEPS} (current=${CURRENT_STEP} + extra=${EXTRA_STEPS})"
echo "[Stage4-Hard-Continue] lr=${LEARNING_RATE}, seg_lr=${SEG_LEARNING_RATE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo "[Stage4-Hard-Continue] bad_samples: record=${RECORD_BAD_SAMPLES}, threshold=${BAD_SAMPLE_DICE_THRESHOLD}"

python -m deepspeed.launcher.runner --master_port="${MASTER_PORT}" --include localhost:0 segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower_mask "/root/autodl-tmp/pretrained_model/mask2former/model_final_54b88a.pkl" \
    --base_data_path "${HARD_DATA_PATH}" \
    --output_dir "${SAVE_OUTPUT_DIR}" \
    --max_steps "${TARGET_MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --bf16 True \
    --save_total_limit 1 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --max_seg_per_sample "${MAX_SEG_PER_SAMPLE}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --use_attention_loss "${USE_ATTENTION_LOSS}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --record_bad_samples "${RECORD_BAD_SAMPLES}" \
    --bad_sample_dice_threshold "${BAD_SAMPLE_DICE_THRESHOLD}" \
    --train_swin_backbone "${TRAIN_SWIN_BACKBONE}" \
    --swin_trainable_stages "${SWIN_TRAINABLE_STAGES}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
    --lora_r "${LORA_R}" \
    --seg_learning_rate "${SEG_LEARNING_RATE}" \
    --adam_epsilon 1e-6 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --data_ratio "1" \
    --switch_bs 4
