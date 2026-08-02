#!/bin/bash
#SBATCH --job-name=ncc_i
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=../logs/inference_ncc.out
#SBATCH --error=../logs/inference_ncc.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1


python inference_ncc.py

