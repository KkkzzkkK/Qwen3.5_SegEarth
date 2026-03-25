#!/usr/bin/env bash
set -euo pipefail
BASE_DATA_PATH=${BASE_DATA_PATH:-"/root/autodl-tmp/data"}
MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/output4/segearth_r2_merged2"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/output4/eval_result"}
MASTER_PORT=${MASTER_PORT:-29500}

python -m deepspeed.launcher.runner --master_port="${MASTER_PORT}" --include localhost:0 segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_batch_size 3