#!/bin/bash

#SBATCH --job-name=3dg-results-no-updates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

# Run training job: Minitank Adversarial with HAPPO-ADV
python ../paper_results.py \
    --config ../configs/3d_gal_no_updates.yaml