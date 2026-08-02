#!/bin/bash
#SBATCH --job-name=check_ncc
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=../logs/run_metrics.out
#SBATCH --error=../logs/run_metrics.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1


python run_cv_v1.py \
    --identities "ncc,psnr,mi,ssim" \
    --csv ~/benchmarking/filters/train_classification_renamed_columns_2.csv \
    --output results/my_run.pkl

