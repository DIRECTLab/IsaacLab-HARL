#!/bin/bash

#SBATCH --job-name=train-base-anymal-go-to-point-sumo
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

# Run training job: anymal c walking to a point in the sumo arena with HAPPO-ADV
python ../../train.py \
    --algorithm happo_adv \
    --num_envs 5_000 \
    --num_env_steps 1_000_000_000 \
    --task "Anymal-C-Go-To-Point-Sumo" \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_anymal_go_to_point_sumo" \
    --save_checkpoints \
    --checkpoint_interval 200
