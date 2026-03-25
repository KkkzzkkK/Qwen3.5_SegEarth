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

# Stage-3 mix continue script:
# - Resume from latest checkpoint produced by scripts/train_stage2_aug.sh
# - Build a mixed dataset from original + new dataset into an independent root
# - Continue training on the mixed dataset

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

# -------- Runtime args --------
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-$DATALOADER_WORKERS_DEFAULT}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero1.json}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-3072}
QWEN_IMAGE_TOKEN_BUDGET=${QWEN_IMAGE_TOKEN_BUDGET:-1024}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
MAX_SEG_PER_SAMPLE=${MAX_SEG_PER_SAMPLE:-10}
USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
TRAIN_SWIN_BACKBONE=${TRAIN_SWIN_BACKBONE:-True}
SWIN_TRAINABLE_STAGES=${SWIN_TRAINABLE_STAGES:-2,3}
ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME=${ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME:-1}

# -------- Stage control --------
STAGE2_AUG_OUTPUT_DIR=${STAGE2_AUG_OUTPUT_DIR:-/root/autodl-tmp/output2/segearth_r2_lora}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/output3/segearth_r2_lora}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/autodl-tmp/qwen}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-1}

STAGE2_MAX_STEPS=${STAGE2_MAX_STEPS:-7500}
STAGE3_EXTRA_STEPS=${STAGE3_EXTRA_STEPS:-7500}
STAGE3_MAX_STEPS=$((STAGE2_MAX_STEPS + STAGE3_EXTRA_STEPS))

resolve_dataset_root() {
    local input_root="$1"
    local var_name="$2"

    if [ -f "${input_root}/train/annotations/train_data.json" ]; then
        echo "${input_root}"
        return 0
    fi

    local candidates=(
        "${input_root}/data"
        "${input_root}/data_aug_lite"
        "${input_root}/geopixeld_stage"
        "${input_root}/geopixeld_stage2"
        "${input_root}/data_mix_stage3"
    )

    for cand in "${candidates[@]}"; do
        if [ -f "${cand}/train/annotations/train_data.json" ]; then
            echo "[Stage3-Mix] ${var_name} auto-resolved to: ${cand}" >&2
            echo "${cand}"
            return 0
        fi
    done

    echo "[ERROR] ${var_name} is invalid: ${input_root}" >&2
    echo "Expected train_data.json at ${input_root}/train/annotations/train_data.json" >&2
    echo "Also tried common subdirs: data, data_aug_lite, geopixeld_stage, geopixeld_stage2, data_mix_stage3" >&2
    return 1
}

# -------- Mix dataset control --------
BASE_DATA_PATH_ORIG=${BASE_DATA_PATH_ORIG:-/root/autodl-tmp/data_aug_lite}
BASE_DATA_PATH_NEW=${BASE_DATA_PATH_NEW:-/root/autodl-tmp/geopixeld_stage}
BASE_DATA_PATH_MIX=${BASE_DATA_PATH_MIX:-/root/autodl-tmp/data_mix_stage3}
AUTO_MIX_DATASET=${AUTO_MIX_DATASET:-1}
MIX_KEEP_RATIO_ORIG=${MIX_KEEP_RATIO_ORIG:-0.2}
MIX_KEEP_RATIO_NEW=${MIX_KEEP_RATIO_NEW:-1.0}
MIX_SEED=${MIX_SEED:-42}

BASE_DATA_PATH_ORIG="$(resolve_dataset_root "${BASE_DATA_PATH_ORIG}" "BASE_DATA_PATH_ORIG")"
BASE_DATA_PATH_NEW="$(resolve_dataset_root "${BASE_DATA_PATH_NEW}" "BASE_DATA_PATH_NEW")"

TRAIN_BASE_DATA_PATH="${BASE_DATA_PATH_MIX}"

if [ ! -f "${TRAIN_BASE_DATA_PATH}/train/annotations/train_data.json" ]; then
    if [ "${AUTO_MIX_DATASET}" = "1" ] || [ "${AUTO_MIX_DATASET}" = "true" ]; then
        if [ ! -d "${BASE_DATA_PATH_ORIG}" ]; then
            echo "[ERROR] BASE_DATA_PATH_ORIG does not exist: ${BASE_DATA_PATH_ORIG}"
            exit 1
        fi
        if [ ! -d "${BASE_DATA_PATH_NEW}" ]; then
            echo "[ERROR] BASE_DATA_PATH_NEW does not exist: ${BASE_DATA_PATH_NEW}"
            exit 1
        fi

        echo "[Stage3-Mix] AUTO_MIX_DATASET enabled. Preparing mixed dataset..."
        "${PYTHON_BIN}" scripts/prepare_mix_dataset.py \
            --src-a "${BASE_DATA_PATH_ORIG}" \
            --src-b "${BASE_DATA_PATH_NEW}" \
            --dst "${BASE_DATA_PATH_MIX}" \
            --keep-ratio-a "${MIX_KEEP_RATIO_ORIG}" \
            --keep-ratio-b "${MIX_KEEP_RATIO_NEW}" \
            --seed "${MIX_SEED}" \
            --overwrite 1
    else
        echo "[ERROR] Mixed train_data.json not found: ${TRAIN_BASE_DATA_PATH}/train/annotations/train_data.json"
        echo "Set AUTO_MIX_DATASET=1 or prepare BASE_DATA_PATH_MIX manually."
        exit 1
    fi
fi

LATEST_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [ -n "${LATEST_CHECKPOINT}" ] && [ ! -d "${LATEST_CHECKPOINT}" ]; then
    echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: ${LATEST_CHECKPOINT}"
    exit 1
fi

CHECKPOINT_SEARCH_DIRS=("${STAGE2_AUG_OUTPUT_DIR}")
if [ "${OUTPUT_DIR}" != "${STAGE2_AUG_OUTPUT_DIR}" ]; then
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
    exit 1
fi

export RESUME_FROM_CHECKPOINT="${LATEST_CHECKPOINT}"
export ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME
export QWEN_IMAGE_TOKEN_BUDGET

MODEL_NAME_OR_PATH="${QWEN_MODEL_PATH}"
TRAINER_RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}"
if [ "${RESUME_MODEL_ONLY}" = "1" ] || [ "${RESUME_MODEL_ONLY}" = "true" ]; then
    if [ -f "${RESUME_FROM_CHECKPOINT}/adapter_config.json" ]; then
        MODEL_NAME_OR_PATH="${RESUME_FROM_CHECKPOINT}"
        TRAINER_RESUME_FROM_CHECKPOINT=""
    else
        echo "[WARN] RESUME_MODEL_ONLY requested but adapter_config.json is missing in checkpoint; fallback to trainer-state resume."
    fi
fi

if [ -n "${TRAINER_RESUME_FROM_CHECKPOINT}" ]; then
    export RESUME_FROM_CHECKPOINT="${TRAINER_RESUME_FROM_CHECKPOINT}"
else
    unset RESUME_FROM_CHECKPOINT
fi

LORA_R=${LORA_R:-}
if [ -z "${LORA_R}" ]; then
    ADAPTER_CONFIG_PATH="${MODEL_NAME_OR_PATH}/adapter_config.json"
    if [ -f "${ADAPTER_CONFIG_PATH}" ]; then
        LORA_R=$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("r", 16))' "${ADAPTER_CONFIG_PATH}")
    else
        LORA_R=16
    fi
fi

echo "[Stage3-Mix] Checkpoint source: ${LATEST_CHECKPOINT}"
echo "[Stage3-Mix] Resume mode: $([ -n "${TRAINER_RESUME_FROM_CHECKPOINT}" ] && echo trainer_state || echo model_only)"
echo "[Stage3-Mix] model_name_or_path: ${MODEL_NAME_OR_PATH}"
echo "[Stage3-Mix] Output dir: ${OUTPUT_DIR}"
echo "[Stage3-Mix] Mixed data path: ${TRAIN_BASE_DATA_PATH}"
echo "[Stage3-Mix] mix keep ratios: orig=${MIX_KEEP_RATIO_ORIG}, new=${MIX_KEEP_RATIO_NEW}"
echo "[Stage3-Mix] LoRA rank: ${LORA_R}"
echo "[Stage3-Mix] Attention impl: ${ATTN_IMPLEMENTATION}; attention loss: ${USE_ATTENTION_LOSS}"
echo "[Stage3-Mix] Swin train backbone: ${TRAIN_SWIN_BACKBONE}; trainable stages: ${SWIN_TRAINABLE_STAGES}"
echo "[Stage3-Mix] model_max_length: ${MODEL_MAX_LENGTH}; qwen_image_token_budget: ${QWEN_IMAGE_TOKEN_BUDGET}"
echo "[Stage3-Mix] max_seg_per_sample: ${MAX_SEG_PER_SAMPLE}"
echo "[Stage3-Mix] max_steps: ${STAGE3_MAX_STEPS} (stage2=${STAGE2_MAX_STEPS} + extra=${STAGE3_EXTRA_STEPS})"

python -m deepspeed.launcher.runner --master_port="${MASTER_PORT}" --include localhost:0 segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower_mask "/root/autodl-tmp/pretrained_model/mask2former/model_final_54b88a.pkl" \
    --base_data_path "${TRAIN_BASE_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${STAGE3_MAX_STEPS}" \
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
    --max_seg_per_sample "${MAX_SEG_PER_SAMPLE}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --use_attention_loss "${USE_ATTENTION_LOSS}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --train_swin_backbone "${TRAIN_SWIN_BACKBONE}" \
    --swin_trainable_stages "${SWIN_TRAINABLE_STAGES}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
    --lora_r "${LORA_R}" \
    --seg_learning_rate 1e-4 \
    --adam_epsilon 1e-6 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --data_ratio "1" \
    --switch_bs 4
