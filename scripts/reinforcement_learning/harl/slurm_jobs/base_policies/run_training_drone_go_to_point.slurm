#!/bin/bash

#SBATCH --job-name=train-base-drone-go-to-point
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=nvidia:1
#SBATCH --mem=32G
#SBATCH --output=./logs/%x-%j.log
#SBATCH --error=./logs/%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

set -eo pipefail

# Run training job: crazyflie drone flying to a point with HAPPO
python ../../train.py \
    --algorithm happo \
    --num_envs 10_000 \
    --num_env_steps 1_000_000_000 \
    --task "Isaac-Quadcopter-Direct-HARL-v0" \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_drone_go_to_point" \
    --save_checkpoints \
    --checkpoint_interval 200
