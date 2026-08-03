#!/bin/bash
#SBATCH --job-name=test_i
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=../logs/inference_test_ext.out
#SBATCH --error=../logs/inference_test_ext.err


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

# Размеры для spatial-режимов
OUTPUT_SIZE_FULL=1024
OUTPUT_SIZE_DOWNSAMPLE=512
PATCH_SIZE=256

# Директория для результатов
OUTPUT_DIR="./results"
mkdir -p "${OUTPUT_DIR}"


# --------------------------------------------------
# Вспомогательная функция запуска
# --------------------------------------------------

run_metric() {
    local METRIC=$1
    local SPATIAL=$2
    local DATASET=$3
    local OUTPUT_SIZE=$4
    local OUTPUT_CSV=$5

    EXTRA_ARGS=""

    if [ "${DATASET}" == "mist" ]; then
        EXTRA_ARGS="--source-mpp ${SOURCE_MPP} --target-mpp ${TARGET_MPP}"
    fi

    if [ "${METRIC}" == "lpips" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --lpips-net vgg"
    fi

    echo ""
    echo ">>> ${METRIC^^} | dataset=${DATASET} | spatial_mode=${SPATIAL} | output_size=${OUTPUT_SIZE}"

    python calculate_metric.py \
        --metric "${METRIC}" \
        --csv "${CSV_PATH}" \
        --output "${OUTPUT_CSV}" \
        --device "${DEVICE}" \
        --dataset "${DATASET}" \
        --spatial-mode "${SPATIAL}" \
        --output-size "${OUTPUT_SIZE}" \
        --patch-size "${PATCH_SIZE}" \
        ${EXTRA_ARGS}

    echo ">>> Done: ${OUTPUT_CSV}"
}


# --------------------------------------------------
# Прогоны
# --------------------------------------------------

for METRIC in ncc lpips; do
    for DATASET in mist generic; do

        echo ""
        echo "============================================="
        echo " Metric: ${METRIC^^} | Dataset: ${DATASET}"
        echo "============================================="

        for SPATIAL in full patches; do
            OUTPUT_CSV="${OUTPUT_DIR}/${DATASET}_${METRIC}_${SPATIAL}.csv"
            run_metric "${METRIC}" "${SPATIAL}" "${DATASET}" "${OUTPUT_SIZE_FULL}" "${OUTPUT_CSV}"
        done

        # downsample — отдельный output_size
        OUTPUT_CSV="${OUTPUT_DIR}/${DATASET}_${METRIC}_downsample.csv"
        run_metric "${METRIC}" "downsample" "${DATASET}" "${OUTPUT_SIZE_DOWNSAMPLE}" "${OUTPUT_CSV}"

    done
done


echo ""
echo "============================================="
echo " All done! Results saved in ${OUTPUT_DIR}/"
echo "============================================="
