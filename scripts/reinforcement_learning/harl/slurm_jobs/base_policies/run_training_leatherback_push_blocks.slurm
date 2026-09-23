#!/bin/bash

#SBATCH --job-name=train-base-leatherback-push-blocks
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

# Run training job: leatherback pushing blocks out of the sumo arena with HAPPO-ADV
python ../../train.py \
    --algorithm happo_adv \
    --num_envs 5_000 \
    --num_env_steps 1_000_000_000 \
    --task "leatherback-Sumo-Direct-MA-Stage1-v0" \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_leatherback_push_blocks" \
    --save_checkpoints \
    --checkpoint_interval 200
