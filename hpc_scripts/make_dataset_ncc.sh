#!/bin/bash
#SBATCH --job-name=filter_ncc_patches
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=../logs/make_filtered_datasets_ncc_patches.out
#SBATCH --error=../logs/make_filtered_datasets_ncc_patches.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1

echo "============================================="
echo " FILTERING TRAIN DATASET"
echo " Metric: NCC patches (best AUC)"
echo " AUC_3class=0.7281, AUC_2class=0.7632"
echo "============================================="

# NCC patches - удаляем 15% худших с train выборки
echo ""
echo "=== Filtering 15% worst from TRAIN (NCC patches) ==="

python make_filtered_dataset.py \
    --csv ~/benchmarking/results/mist_ncc_patches.csv \
    --source ~/benchmarking/datasets/MIST_HER2 \
    --target ~/benchmarking/datasets/MIST_HER2_NCC_PATCHES_15 \
    --score-column ncc_mist_mpp1.0112_patches \
    --remove-percent 15

# NCC patches - удаляем 25% худших с train выборки
echo ""
echo "=== Filtering 25% worst from TRAIN (NCC patches) ==="

python make_filtered_dataset.py \
    --csv ~/benchmarking/results/mist_ncc_patches.csv \
    --source ~/benchmarking/datasets/MIST_HER2 \
    --target ~/benchmarking/datasets/MIST_HER2_NCC_PATCHES_25 \
    --score-column ncc_mist_mpp1.0112_patches \
    --remove-percent 25

echo ""
echo "============================================="
echo " Done! Filtered datasets created."
echo "============================================="