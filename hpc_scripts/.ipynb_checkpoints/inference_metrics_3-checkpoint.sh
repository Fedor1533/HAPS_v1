#!/bin/bash
#SBATCH --job-name=test_i
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=../logs/inference_test_3.out
#SBATCH --error=../logs/inference_test_3.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1

CSV_PATH="mist_test_classification_balanced.csv"
DEVICE="cuda"

# MPP параметры
SOURCE_MPP=0.4661
TARGET_MPP=1.0112

# Директория для результатов
OUTPUT_DIR="./results"
mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo " Starting metric calculation for MIST dataset"
echo " Source MPP: ${SOURCE_MPP}"
echo " Target MPP: ${TARGET_MPP}"
echo "============================================="

# --------------------------------------------------
# Вспомогательная функция запуска
# --------------------------------------------------

run_metric() {
    local METRIC=$1
    local DATASET=$2
    local SPATIAL=$3
    local OUTPUT_CSV=$4

    EXTRA_ARGS=""

    if [ "${DATASET}" == "mist" ]; then
        EXTRA_ARGS="--source-mpp ${SOURCE_MPP} --target-mpp ${TARGET_MPP}"
    fi

    if [ "${METRIC}" == "lpips" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --lpips-net vgg"
    fi

    echo ""
    echo ">>> ${METRIC^^} | dataset=${DATASET} | spatial_mode=${SPATIAL}"

    python calculate_metric.py \
        --metric "${METRIC}" \
        --csv "${CSV_PATH}" \
        --output "${OUTPUT_CSV}" \
        --device "${DEVICE}" \
        --dataset "${DATASET}" \
        --spatial-mode "${SPATIAL}" \
        ${EXTRA_ARGS}

    echo ">>> Done: ${OUTPUT_CSV}"
}

# --------------------------------------------------
# Прогоны
# --------------------------------------------------
#
# 1) 1024x1024                        → generic + full
# 2) 1024x1024 разбитое на патчи      → generic + patches
# 3) Задаунсемпленное через MPP        → mist    + downsample
# 4) Задаунсемпленное + патчи          → mist    + patches
#
# --------------------------------------------------

for METRIC in ncc lpips; do

    echo ""
    echo "============================================="
    echo " Metric: ${METRIC^^}"
    echo "============================================="

    # 1) Оригинал 1024x1024
    run_metric "${METRIC}" "generic" "full" \
        "${OUTPUT_DIR}/${METRIC}_original_full.csv"

    # 2) Оригинал 1024x1024, разбитый на патчи
    run_metric "${METRIC}" "generic" "patches" \
        "${OUTPUT_DIR}/${METRIC}_original_patches.csv"

    # 3) Задаунсемпленное через MPP
    run_metric "${METRIC}" "mist" "downsample" \
        "${OUTPUT_DIR}/${METRIC}_mpp_downsample.csv"

    # 4) Задаунсемпленное через MPP, разбитое на патчи
    run_metric "${METRIC}" "mist" "patches" \
        "${OUTPUT_DIR}/${METRIC}_mpp_patches.csv"

done

echo ""
echo "============================================="
echo " All done! Results saved in ${OUTPUT_DIR}/"
echo "============================================="
