#!/bin/bash

#SBATCH --job-name=train-3d-galaga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

# Run training job: Minitank Adversarial with HAPPO-ADV
# set adversarial training iterations to a large number to effectively disable CF updates and only perform policy updates for tank
python ../train.py \
    --algorithm happo_adv \
    --num_envs 10_000 \
    --num_env_steps 10_000_000_000 \
    --headless \
    --task Minitank-Adversarial-Direct-v0 \
    --load_starting_policy \
    --save_interval 5 \
    --log_interval 1 \
    --adversarial_training_mode leapfrog \
    --save_checkpoints \
    --exp_name "3d_galaga_no_update" \
    --checkpoint_interval 200 \
    --adversarial_training_iterations 10_000_000_000