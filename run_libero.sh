#!/usr/bin/env bash
set -euo pipefail

# Standard LIBERO training recipe for the public repository.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LIBERO_ROOT="${LIBERO_ROOT:-../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-data/libero_config}"
export VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --task_suite goal \
  --model_mode libero \
  --use_gte_encoder False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 200000 \
  --max_steps 200005 \
  --save_freq 2000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "wandb" \
  --wandb_project "vla_adapter"
