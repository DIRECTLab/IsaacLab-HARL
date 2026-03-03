#!/bin/bash

#SBATCH --job-name=3dg-results-drone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --output=./logs/%x-%j.log
#SBATCH --error=./logs/%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

# Run training job: Minitank Adversarial with HAPPO-ADV
python ../paper_results.py \
    --config ../configs/3d_gal_adversarial_drone.yaml
