#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export LIBEROPLUS_ROOT="${LIBEROPLUS_ROOT:-../LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-data/liberoplus_config}"
export VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
PROMPT_ENCODER_PATH="${PROMPT_ENCODER_PATH:-Alibaba-NLP/gte-Qwen2-1.5B-instruct}"

$CONDA_PREFIX/bin/python experiments/robot/libero/run_liberoplus_eval.py \
  --prompt_encoder_path "$PROMPT_ENCODER_PATH" \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/goal/goal_lora_320--30000_chkpt \
  --task_suite_name libero_goal \
  --use_pro_version True \
  --task_category "Camera Viewpoints" \
  --num_trials_per_task 1
