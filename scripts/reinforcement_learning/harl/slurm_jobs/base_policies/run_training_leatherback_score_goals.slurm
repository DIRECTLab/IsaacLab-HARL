#!/bin/bash

#SBATCH --job-name=train-base-leatherback-score-goals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=nvidia:1
#SBATCH --mem=24G
#SBATCH --output=./logs/%x-%j.log
#SBATCH --error=./logs/%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

set -eo pipefail

# Run training job: leatherback scoring goals with HAPPO
python ../../train.py \
    --algorithm happo \
    --num_envs 5_000 \
    --num_env_steps 1_000_000_000 \
    --task "Leatherback-Stage1-Soccer-v0" \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_leatherback_score_goals" \
    --save_checkpoints \
    --checkpoint_interval 200
