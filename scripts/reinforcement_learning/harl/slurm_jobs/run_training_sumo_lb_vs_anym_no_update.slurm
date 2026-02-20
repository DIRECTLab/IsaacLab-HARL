#!/bin/bash

#SBATCH --job-name=train-sumo-lb-vs-anym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

# Run training job: leatherback team vs anymal team sumo with HAPPO-ADV, no update on team 1 (leatherback)
python ../train.py \
    --algorithm happo_adv \
    --num_envs 10_000 \
    --num_env_steps 10_000_000_000 \
    --headless \
    --task "Sumo-Stage2-Hetero-By-Team-v0" \
    --load_starting_policy \
    --save_interval 5 \
    --log_interval 1 \
    --adversarial_training_mode leapfrog \
    --exp_name "no_update_sumo_lb_vs_anym_happo_adv" \
    --save_checkpoints \
    --checkpoint_interval 200 \
    --adversarial_training_iterations 10_000_000_000
