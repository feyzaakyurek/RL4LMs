#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -N t5largescript           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -o /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric/t5large_scripting.log
#$ -m bea             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -t 1

module load conda
conda activate rl4lm

if [[ 1 -eq $SGE_TASK_ID ]]; then
BASE_PATH="/projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric"
PROJECT_NAME="rl4lm_exps"
EXPERIMENT_NAME="edit_numeric_t5_large_base_me"
mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME
WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised_1.yml \
--base_path_to_store_results $BASE_PATH \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--log_to_wandb > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/$EXPERIMENT_NAME.log 2>&1
fi

# if [[ 2 -eq $SGE_TASK_ID ]]; then
# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised_2.yml \
# --base_path_to_store_results /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric \
# --project_name "rl4lm_exps" \
# --experiment_name "edit_numeric_t5_large_topk200_hub" \
# --log_to_wandb > /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric/edit_numeric_t5_large_topk200_hub.log 2>&1
# fi

# if [[ 3 -eq $SGE_TASK_ID ]]; then
# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised_3.yml \
# --base_path_to_store_results /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric \
# --project_name "rl4lm_exps" \
# --experiment_name "edit_numeric_t5_large_lr10e6_der" \
# --log_to_wandb > /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric/edit_numeric_t5_large_lr10e6_der.log 2>&1
# fi

# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised_1.yml \
# --base_path_to_store_results /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric \
# --project_name "rl4lm_exps" \
# --experiment_name "edit_numeric_t5_large_base_me_debug"

# if [[ 1 -eq $SGE_TASK_ID ]]; then
# mkdir -p /projectnb/xxxxx/feyzanb/feedback/interscript_osssutput/edit_numeric/rl4lm_exps/edit_numeric_t5_large_base_me_bs32
# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --config_path scripts/training/task_configs/scripting/t5_ppo_on_supervised_1.yml \
# --base_path_to_store_results /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric \
# --project_name "rl4lm_exps" \
# --experiment_name "edit_numeric_t5_large_base_hub_gpu2" \
# --log_to_wandb > /projectnb/llamagrp/feyzanb/feedback/interscript_output/edit_numeric/rl4lm_exps/edit_numeric_t5_large_base_hub_gpu2/out.log 2>&1
# fi
