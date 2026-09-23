#!/bin/bash

#SBATCH --job-name=train-base-anymal-push-blocks
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

# Stage the starting policy from the previous base-policy run
./stage_policy.sh ./starting_policies/anymal_push_blocks \
    Anymal-C-Go-To-Point-Sumo happo_adv base_anymal_go_to_point_sumo \
    robot_0:robot_0 robot_0:robot_1

# Run training job: anymal c pushing blocks out of the sumo arena with HAPPO-ADV, starting from go-to-point sumo
python ../../train.py \
    --algorithm happo_adv \
    --num_envs 5_000 \
    --num_env_steps 1_000_000_000 \
    --task "Anymal-C-Sumo-Stage1-Blocks-Push-v0" \
    --dir ./starting_policies/anymal_push_blocks \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_anymal_push_blocks" \
    --save_checkpoints \
    --checkpoint_interval 200
