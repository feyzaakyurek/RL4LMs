#!/bin/sh

# mkdir -p /output/t5

# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised.yml \
# --base_path_to_store_results /output/t5 \
# --log_to_wandb

mkdir -p /output/t5

WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised.yml \
--base_path_to_store_results /output/t5 \
--log_to_wandb

# mkdir -p ./output/t5

# python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised.yml \
# --base_path_to_store_results ./output/t5
