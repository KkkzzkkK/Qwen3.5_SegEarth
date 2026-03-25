#!/usr/bin/env bash
set -euo pipefail

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

# Stage-2 training script:
# - Resume from Stage-1 checkpoint
# - Train one more round with (light) augmented data
# - Keep scripts/train.sh unchanged

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CPU_CORES=$(nproc)
if [ "$CPU_CORES" -gt 10 ]; then
    DATALOADER_WORKERS_DEFAULT=10
else
    DATALOADER_WORKERS_DEFAULT=$CPU_CORES
fi

MASTER_PORT=${MASTER_PORT:-29500}

# -------- Shared runtime args --------
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-$DATALOADER_WORKERS_DEFAULT}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero1.json}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME=${ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME:-1}

# -------- Stage-1 / Stage-2 control --------
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-/root/autodl-tmp/output/segearth_r2_lora}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/output2/segearth_r2_lora}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/autodl-tmp/qwen}
STAGE1_MAX_STEPS=${STAGE1_MAX_STEPS:-5000}
STAGE2_EXTRA_STEPS=${STAGE2_EXTRA_STEPS:-2500}
STAGE2_MAX_STEPS=$((STAGE1_MAX_STEPS + STAGE2_EXTRA_STEPS))

# Use augmented dataset directory for stage-2.
# Example: /root/autodl-tmp/data_aug_lite
BASE_DATA_PATH=${BASE_DATA_PATH:-/root/autodl-tmp}
BASE_DATA_PATH_AUG=${BASE_DATA_PATH_AUG:-/root/autodl-tmp/data_aug_lite}
AUTO_AUGMENT=${AUTO_AUGMENT:-1}
AUG_RATIO=${AUG_RATIO:-0.3}
AUG_SEED=${AUG_SEED:-42}

TRAIN_BASE_DATA_PATH="${BASE_DATA_PATH_AUG}"

if [ ! -d "${TRAIN_BASE_DATA_PATH}" ]; then
    if [ "${AUTO_AUGMENT}" = "1" ] || [ "${AUTO_AUGMENT}" = "true" ]; then
        if [ ! -d "${BASE_DATA_PATH}" ]; then
            echo "[ERROR] BASE_DATA_PATH does not exist: ${BASE_DATA_PATH}"
            echo "Cannot auto-generate augmented data without the stage-1 dataset root."
            exit 1
        fi
        echo "[Stage2] AUTO_AUGMENT enabled. Preparing lite augmented data..."
        "${PYTHON_BIN}" scripts/prepare_lite_aug.py \
            --src "${BASE_DATA_PATH}" \
            --dst "${BASE_DATA_PATH_AUG}" \
            --ratio "${AUG_RATIO}" \
            --seed "${AUG_SEED}" \
            --overwrite 1
    else
        echo "[ERROR] BASE_DATA_PATH_AUG does not exist: ${BASE_DATA_PATH_AUG}"
        echo "Prepare augmented data first, or rerun with AUTO_AUGMENT=1."
        exit 1
    fi
fi

if [ ! -f "${TRAIN_BASE_DATA_PATH}/train/annotations/train_data.json" ]; then
    echo "[ERROR] train_data.json not found under ${TRAIN_BASE_DATA_PATH}/train/annotations"
    echo "Set BASE_DATA_PATH_AUG to a valid dataset root, or regenerate with AUTO_AUGMENT=1."
    exit 1
fi

LATEST_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [ -n "${LATEST_CHECKPOINT}" ] && [ ! -d "${LATEST_CHECKPOINT}" ]; then
    echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: ${LATEST_CHECKPOINT}"
    exit 1
fi

CHECKPOINT_SEARCH_DIRS=("${STAGE1_OUTPUT_DIR}")
if [ "${OUTPUT_DIR}" != "${STAGE1_OUTPUT_DIR}" ]; then
    CHECKPOINT_SEARCH_DIRS+=("${OUTPUT_DIR}")
fi

if [ -z "${LATEST_CHECKPOINT}" ]; then
    for CHECKPOINT_SEARCH_DIR in "${CHECKPOINT_SEARCH_DIRS[@]}"; do
        if ls -d "${CHECKPOINT_SEARCH_DIR}"/checkpoint-* >/dev/null 2>&1; then
            LATEST_CHECKPOINT="$(ls -d "${CHECKPOINT_SEARCH_DIR}"/checkpoint-* | sort -V | tail -n 1)"
            break
        fi
    done
fi

if [ -z "${LATEST_CHECKPOINT}" ]; then
    echo "[ERROR] No checkpoint found in any of these locations:"
    for CHECKPOINT_SEARCH_DIR in "${CHECKPOINT_SEARCH_DIRS[@]}"; do
        echo "  - ${CHECKPOINT_SEARCH_DIR}/checkpoint-*"
    done
    echo "Run stage-1 training first (scripts/train.sh), or set STAGE1_OUTPUT_DIR / RESUME_FROM_CHECKPOINT explicitly."
    exit 1
fi

export RESUME_FROM_CHECKPOINT="${LATEST_CHECKPOINT}"
export ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME

LORA_R=${LORA_R:-}
if [ -z "${LORA_R}" ]; then
    ADAPTER_CONFIG_PATH="${RESUME_FROM_CHECKPOINT}/adapter_config.json"
    if [ -f "${ADAPTER_CONFIG_PATH}" ]; then
        LORA_R=$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("r", 16))' "${ADAPTER_CONFIG_PATH}")
    else
        LORA_R=16
    fi
fi

echo "[Stage2] Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "[Stage2] Output dir: ${OUTPUT_DIR}"
echo "[Stage2] Data path: ${TRAIN_BASE_DATA_PATH}"
echo "[Stage2] LoRA rank: ${LORA_R}"
echo "[Stage2] Attention impl: ${ATTN_IMPLEMENTATION}; attention loss: ${USE_ATTENTION_LOSS}"
echo "[Stage2] max_steps: ${STAGE2_MAX_STEPS} (stage1=${STAGE1_MAX_STEPS} + extra=${STAGE2_EXTRA_STEPS})"

python -m deepspeed.launcher.runner --master_port="${MASTER_PORT}" --include localhost:0 segearth_r2/train/train.py \
    --model_name_or_path "${QWEN_MODEL_PATH}" \
    --vision_tower_mask "/root/autodl-tmp/pretrained_model/mask2former/model_final_54b88a.pkl" \
    --base_data_path "${TRAIN_BASE_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${STAGE2_MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --save_strategy "steps" \
    --save_steps 200 \
    --bf16 True \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --use_attention_loss "${USE_ATTENTION_LOSS}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
    --lora_r "${LORA_R}" \
    --seg_learning_rate 1e-4 \
    --adam_epsilon 1e-6 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --data_ratio "1" \
    --switch_bs 4
