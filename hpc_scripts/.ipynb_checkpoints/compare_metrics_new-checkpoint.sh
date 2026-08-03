#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=../logs/compare_metrics_new.out
#SBATCH --error=../logs/compare_metrics_new.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1

OUTPUT_DIR="${HOME}/benchmarking/HAPS_v1/results"
ANALYSIS_DIR="${HOME}/benchmarking/HAPS_v1/analysis"
MERGED_CSV="${OUTPUT_DIR}/all_metrics_merged.csv"

# Пересоздаём merged файл всегда
echo "Создаю объединённый файл метрик..."

for METRIC in ncc lpips; do
    for EXPERIMENT in original_full original_patches mpp_downsample mpp_patches; do
        FILE="${OUTPUT_DIR}/${METRIC}_${EXPERIMENT}.csv"
        if [ -f "${FILE}" ]; then
            echo "  Найден: $(basename ${FILE})"
        else
            echo "  Не найден: $(basename ${FILE})"
        fi
    done
done

python merge_metrics.py \
    --results-dir "${OUTPUT_DIR}" \
    --output "${MERGED_CSV}"

if [ $? -ne 0 ]; then
    echo "Ошибка при создании объединённого файла"
    exit 1
fi

echo "Анализ файла: ${MERGED_CSV}"

python analyze_metrics.py \
    --csv "${MERGED_CSV}" \
    --output-dir "${ANALYSIS_DIR}"

echo "Готово. Результаты в: ${ANALYSIS_DIR}"
