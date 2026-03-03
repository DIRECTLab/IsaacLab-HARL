#!/bin/bash

#SBATCH --job-name=lb-vs-an-results-no-updates
#SBATCH --nodes=1
#SBATCH --nodelist=ryu
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
    --config ../configs/lb_vs_an_soccer_no_updates.yaml
