#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -N t5largesumm           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -o /projectnb/llamagrp/feyzanb/feedback/openai_summ_output/t5large_openai_summ_QMQ3V/t5large_summ.log
#$ -m bea             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 80GB of GPU memory per GPU
#$ -t 1-2

module load conda
conda activate rl4lm

if [[ 1 -eq $SGE_TASK_ID ]]; then
BASE_PATH="/projectnb/llamagrp/feyzanb/feedback/openai_summ_output"
PROJECT_NAME="rl4lm_exps"
EXPERIMENT_NAME="t5large_QMQ3V_bs2_lr10e6_ent_coef10e3_envs10_hub"
mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME
WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_ppo_on_supervised_1.yml
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--log_to_wandb > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/$EXPERIMENT_NAME.log 2>&1
fi

if [[ 2 -eq $SGE_TASK_ID ]]; then
BASE_PATH="/projectnb/llamagrp/feyzanb/feedback/openai_summ_output"
PROJECT_NAME="rl4lm_exps"
EXPERIMENT_NAME="t5large_QMQ3V_bs2_lr10e6_ent_coef10e3_envs20_der"
mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME
WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_ppo_on_supervised_2.yml
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--log_to_wandb > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/$EXPERIMENT_NAME.log 2>&1
fi

# BASE_PATH="/projectnb/llamagrp/feyzanb/feedback/openai_summ_output"
# PROJECT_NAME="rl4lm_exps"
# EXPERIMENT_NAME="t5large_QMQ3V_bs2_lr10e6_ent_coef10e3_hub"
# WANDB_API_KEY=2b7a2ef5bf341cde3a0948d661795ec96c04a0a1 python scripts/training/train_text_generation.py \
# --base_path_to_store_results $BASE_PATH \
# --config_path scripts/training/task_configs/openai_summ/t5_ppo_on_supervised_1.yml
# --project_name $PROJECT_NAME \
# --experiment_name $EXPERIMENT_NAME
