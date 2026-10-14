#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LIBEROPLUS_ROOT="${LIBEROPLUS_ROOT:-../LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-data/liberoplus_config}"
export VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
PROMPT_ENCODER_PATH="${PROMPT_ENCODER_PATH:-Alibaba-NLP/gte-Qwen2-1.5B-instruct}"

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--task_suite object \
--model_mode liberoplus \
--use_gte_encoder True \
--prompt_encoder_path "$PROMPT_ENCODER_PATH" \
--prompt_pad_length_multiplier 4 \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 16 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "wandb" \
--wandb_project "vla_adapter" \
--use_visual_token_perturbation True \
--visual_token_pert_prob 0.8 \
--visual_token_pert_full_shuffle_prob 0.1 \
--visual_token_pert_num_blocks_min 3 \
--visual_token_pert_num_blocks_max 6 \
--visual_token_pert_block_size_min 3 \
--visual_token_pert_block_size_max 7
