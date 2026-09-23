#!/bin/bash

#SBATCH --job-name=train-base-anymal-go-to-ball
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
./stage_policy.sh ./starting_policies/anymal_go_to_ball \
    AnymalC_Soccer_Go_To_Point_Stage_0 happo_adv base_anymal_go_to_point_soccer \
    robot_0:robot_0

# Run training job: anymal c walking to the soccer ball with HAPPO, starting from go-to-point soccer
python ../../train.py \
    --algorithm happo \
    --num_envs 5_000 \
    --num_env_steps 1_000_000_000 \
    --task "AnymalC_Soccer_Go_To_Ball_Stage_1" \
    --dir ./starting_policies/anymal_go_to_ball \
    --save_interval 5 \
    --log_interval 1 \
    --headless \
    --exp_name "base_anymal_go_to_ball" \
    --save_checkpoints \
    --checkpoint_interval 200
