#!/bin/bash
#SBATCH --job-name=gen_lpips
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=../logs/make_filtered_datasets_lpips.out
#SBATCH --error=../logs/make_filtered_datasets_lpips.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1

echo "Filtering 15%"

# python make_filtered_dataset.py \
#     --csv ~/benchmarking/HAPS_v1/train_classification_with_ncc.csv \
#     --source ~/benchmarking/datasets/MIST_HER2 \
#     --target ~/benchmarking/datasets/MIST_HER2_NCC_15 \
#     --score-column ncc_hed \
#     --remove-percent 15

# echo "Filtering 25%"

# python make_filtered_dataset.py \
#     --csv ~/benchmarking/HAPS_v1/train_classification_with_ncc.csv \
#     --source ~/benchmarking/datasets/MIST_HER2 \
#     --target ~/benchmarking/datasets/MIST_HER2_NCC_25 \
#     --score-column ncc_hed \
#     --remove-percent 25



python make_filtered_dataset.py \
    --csv ~/benchmarking/HAPS_v1/train_classification_with_lpips.csv \
    --source ~/benchmarking/datasets/MIST_HER2 \
    --target ~/benchmarking/datasets/MIST_HER2_LPIPS_15 \
    --score-column lpips_hed \
    --remove-percent 15

echo "Filtering 25%"

python make_filtered_dataset.py \
    --csv ~/benchmarking/HAPS_v1/train_classification_with_lpips.csv \
    --source ~/benchmarking/datasets/MIST_HER2 \
    --target ~/benchmarking/datasets/MIST_HER2_LPIPS_25 \
    --score-column lpips_hed \
    --remove-percent 25
