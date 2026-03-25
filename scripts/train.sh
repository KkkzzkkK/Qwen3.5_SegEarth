export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CPU_CORES=$(nproc)
if [ "$CPU_CORES" -gt 10 ]; then
    DATALOADER_WORKERS_DEFAULT=12
else
    DATALOADER_WORKERS_DEFAULT=$CPU_CORES
fi

MASTER_PORT=${MASTER_PORT:-29500}
AUTO_DISABLE_SINGLE_GPU_DEEPSPEED=${AUTO_DISABLE_SINGLE_GPU_DEEPSPEED:-1}
FORCE_DEEPSPEED=${FORCE_DEEPSPEED:-1}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-$DATALOADER_WORKERS_DEFAULT}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero1.json}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/output/segearth_r2_lora}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/autodl-tmp/qwen}
RESUME=${RESUME:-auto}
ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME=${ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME:-1}
RECORD_BAD_SAMPLES=${RECORD_BAD_SAMPLES:-True}
BAD_SAMPLE_DICE_THRESHOLD=${BAD_SAMPLE_DICE_THRESHOLD:-40}
BAD_SAMPLE_LOG_FILE=${BAD_SAMPLE_LOG_FILE:-}
BASE_DATA_PATH=${BASE_DATA_PATH:-/root/autodl-tmp/data}
MAX_STEPS=${MAX_STEPS:-7500}
SAVE_STEPS=${SAVE_STEPS:-200}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-300}
DATA_RATIO=${DATA_RATIO:-1}
SWITCH_BS=${SWITCH_BS:-4}

LATEST_CHECKPOINT=""
if ls -d "${OUTPUT_DIR}"/checkpoint-* >/dev/null 2>&1; then
    LATEST_CHECKPOINT="$(ls -d "${OUTPUT_DIR}"/checkpoint-* | sort -V | tail -n 1)"
fi

if [ -z "${RESUME_FROM_CHECKPOINT:-}" ]; then
    if [ "${RESUME}" = "1" ] || [ "${RESUME}" = "true" ] || [ "${RESUME}" = "auto" -a -n "${LATEST_CHECKPOINT}" ]; then
        if [ -n "${LATEST_CHECKPOINT}" ]; then
            export RESUME_FROM_CHECKPOINT="${LATEST_CHECKPOINT}"
            echo "Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
        elif [ "${RESUME}" != "auto" ]; then
            echo "RESUME is enabled but no checkpoint found under ${OUTPUT_DIR}/checkpoint-*"
            exit 1
        fi
    fi
fi

export ALLOW_UNSAFE_TORCH_LOAD_FOR_RESUME

GPU_COUNT=0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a CUDA_VISIBLE_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#CUDA_VISIBLE_DEVICE_ARRAY[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi

if [ "${GPU_COUNT}" -le 0 ]; then
    GPU_COUNT=1
fi

USE_DEEPSPEED=1
if [ "${FORCE_DEEPSPEED}" = "1" ] || [ "${FORCE_DEEPSPEED}" = "true" ]; then
    USE_DEEPSPEED=1
elif [ "${AUTO_DISABLE_SINGLE_GPU_DEEPSPEED}" = "1" ] || [ "${AUTO_DISABLE_SINGLE_GPU_DEEPSPEED}" = "true" ]; then
    if [ "${GPU_COUNT}" -le 1 ]; then
        USE_DEEPSPEED=0
    fi
fi

TRAIN_ARGS=(
    segearth_r2/train/train.py
    --model_name_or_path "${QWEN_MODEL_PATH}"
    --vision_tower_mask "/root/autodl-tmp/pretrained_model/mask2former/model_final_54b88a.pkl"
    --base_data_path "${BASE_DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save_strategy "steps"
    --save_steps "${SAVE_STEPS}"
    --bf16 True
    --save_total_limit 1
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.
    --warmup_steps "${WARMUP_STEPS}"
    --lr_scheduler_type "cosine"
    --logging_steps 10
    --tf32 True
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --use_attention_loss "${USE_ATTENTION_LOSS}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --record_bad_samples "${RECORD_BAD_SAMPLES}"
    --bad_sample_dice_threshold "${BAD_SAMPLE_DICE_THRESHOLD}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
    --lora_r 16
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    --data_ratio "${DATA_RATIO}"
    --switch_bs "${SWITCH_BS}"
    --seg_learning_rate 1e-4
    --adam_epsilon 1e-6
)

if [ -n "${BAD_SAMPLE_LOG_FILE}" ]; then
    TRAIN_ARGS+=(--bad_sample_log_file "${BAD_SAMPLE_LOG_FILE}")
fi

# ------ main-training ------
# 1轮稳妥版（约1 epoch）：
# - 数据集约10000样本，单卡bs=1时，max_steps=10000约等于1轮
# - 通过梯度累积提高等效batch，减小loss抖动
# - 用warmup_steps替代warmup_ratio（兼容新版本transformers）
# 显存不足时可选择zero3.json或者zero2.json
if [ "${USE_DEEPSPEED}" -eq 1 ]; then
    echo "Launching with DeepSpeed across ${GPU_COUNT} visible GPU(s)."
    python -m deepspeed.launcher.runner --master_port="${MASTER_PORT}" --include localhost:0 \
        "${TRAIN_ARGS[@]}" \
        --deepspeed "${DEEPSPEED_CONFIG}"
else
    echo "Single GPU detected; disabling DeepSpeed launcher/config for better throughput."
    python "${TRAIN_ARGS[@]}"
fi
