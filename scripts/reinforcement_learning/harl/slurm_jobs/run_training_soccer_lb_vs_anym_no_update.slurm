#!/bin/bash

#SBATCH --job-name=train-soccer-lb-vs-anym-no-updates
#SBATCH --nodelist=kuma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --output=./logs/%x-%j.log
#SBATCH --error=./logs/%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

# Run training job: leatherback vs anymal soccer with HAPPO-ADV
python ../train.py \
    --algorithm happo_adv \
    --num_envs 5_000 \
    --num_env_steps 10_000_000_000 \
    --headless \
    --task "AnymalC-VS-Leatherback-Soccer-v0" \
    --load_starting_policy \
    --save_interval 5 \
    --log_interval 1 \
    --adversarial_training_mode leapfrog \
    --exp_name "soccer_lb_vs_anym_happo_adv_no_updates" \
    --save_checkpoints \
    --checkpoint_interval 200 \
    --adversarial_training_iterations 10_000_000_000
