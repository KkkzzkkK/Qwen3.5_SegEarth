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

OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/output4/segearth_r2_lora_hard_data_mix_stage3_out_lora_aug3"}
MODEL_PATH=${MODEL_PATH:-"$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)"}
SAVE_PATH=${SAVE_PATH:-"/root/autodl-tmp/output4/segearth_r2_merged2"}

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "No valid checkpoint found. Please set MODEL_PATH or ensure ${OUTPUT_DIR}/checkpoint-* exists."
    exit 1
fi

LORA_R=${LORA_R:-}
if [[ -z "${LORA_R}" ]]; then
    ADAPTER_CONFIG_PATH="${MODEL_PATH}/adapter_config.json"
    if [[ -f "${ADAPTER_CONFIG_PATH}" ]]; then
        LORA_R=$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("r", 16))' "${ADAPTER_CONFIG_PATH}")
    else
        LORA_R=16
    fi
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${MODEL_PATH}" \
    --vision_tower_mask "/root/autodl-tmp/pretrained_model/mask2former/model_final_54b88a.pkl" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --save_path "${SAVE_PATH}" \
    --lora_r "${LORA_R}"

echo "Merged model saved to: ${SAVE_PATH}"