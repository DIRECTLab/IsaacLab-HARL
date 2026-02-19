#!/bin/bash

#SBATCH --job-name=test-fps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate direct_isaaclab_env

python ./benchmark_fps.py