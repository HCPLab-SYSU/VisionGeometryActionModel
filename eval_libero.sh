#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LIBERO_ROOT="${LIBERO_ROOT:-../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-data/libero_config}"
export VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"

$CONDA_PREFIX/bin/python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/object/object_lora_011--36000_chkpt \
  --task_suite_name libero_object \
  --use_pro_version True \
  --num_trials_per_task 50
