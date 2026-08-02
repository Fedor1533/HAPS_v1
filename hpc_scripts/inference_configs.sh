#!/bin/bash
#SBATCH --job-name=lpips_i
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=../logs/inference_all.out
#SBATCH --error=../logs/inference_all.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1



CSV_PATH="train_classification_renamed_columns_2.csv"
DEVICE="cuda"
DATASET="mist"

# MPP параметры
SOURCE_MPP=0.4661
TARGET_MPP=1.0112

# Размеры для spatial-режимов
OUTPUT_SIZE=1024
PATCH_SIZE=256

# Директория для результатов
OUTPUT_DIR="../results"
mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo " Starting metric calculation for MIST dataset"
echo " Source MPP: ${SOURCE_MPP}"
echo " Target MPP: ${TARGET_MPP}"
echo "============================================="

# --------------------------------------------------
# 1. NCC — все три spatial-режима
# --------------------------------------------------

for SPATIAL in full downsample patches; do
    echo ""
    echo ">>> NCC | spatial_mode=${SPATIAL}"
    
    OUTPUT_CSV="${OUTPUT_DIR}/mist_ncc_${SPATIAL}.csv"
    
    python calculate_metric.py \
        --metric ncc \
        --csv "${CSV_PATH}" \
        --output "${OUTPUT_CSV}" \
        --device "${DEVICE}" \
        --dataset "${DATASET}" \
        --source-mpp "${SOURCE_MPP}" \
        --target-mpp "${TARGET_MPP}" \
        --spatial-mode "${SPATIAL}" \
        --output-size "${OUTPUT_SIZE}" \
        --patch-size "${PATCH_SIZE}"
    
    echo ">>> Done: ${OUTPUT_CSV}"
done

# --------------------------------------------------
# 2. LPIPS — все три spatial-режима
# --------------------------------------------------

for SPATIAL in full downsample patches; do
    echo ""
    echo ">>> LPIPS | spatial_mode=${SPATIAL}"
    
    OUTPUT_CSV="${OUTPUT_DIR}/mist_lpips_${SPATIAL}.csv"
    
    python calculate_metric.py \
        --metric lpips \
        --csv "${CSV_PATH}" \
        --output "${OUTPUT_CSV}" \
        --device "${DEVICE}" \
        --dataset "${DATASET}" \
        --source-mpp "${SOURCE_MPP}" \
        --target-mpp "${TARGET_MPP}" \
        --spatial-mode "${SPATIAL}" \
        --output-size "${OUTPUT_SIZE}" \
        --patch-size "${PATCH_SIZE}" \
        --lpips-net vgg
    
    echo ">>> Done: ${OUTPUT_CSV}"
done

echo ""
echo "============================================="
echo " All done! Results saved in ${OUTPUT_DIR}/"
echo "============================================="

