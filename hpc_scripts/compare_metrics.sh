#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=../logs/compare_metrics.out
#SBATCH --error=../logs/compare_metrics.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
cd ~/benchmarking/HAPS_v1


# quick_analyze.sh
# Быстрый анализ уже посчитанных метрик

OUTPUT_DIR="${HOME}/benchmarking/HAPS_v1/results"
ANALYSIS_DIR="${HOME}/benchmarking/HAPS_v1/analysis"

MERGED_CSV="${OUTPUT_DIR}/all_metrics_merged_test.csv"

# Проверяем, существует ли объединенный файл
if [ ! -f "${MERGED_CSV}" ]; then
    echo "Объединенный файл не найден: ${MERGED_CSV}"
    echo "Пытаюсь создать его из отдельных файлов..."
    
    # Собираем все файлы метрик
    METRIC_FILES=()
    for METRIC in ncc lpips; do
        for SPATIAL in full downsample patches; do
            FILE="${OUTPUT_DIR}/mist_${METRIC}_${SPATIAL}_test.csv"
            if [ -f "${FILE}" ]; then
                METRIC_FILES+=("${FILE}")
            fi
        done
    done
    
    if [ ${#METRIC_FILES[@]} -eq 0 ]; then
        echo "Ошибка: не найдено ни одного файла с метриками в ${OUTPUT_DIR}"
        exit 1
    fi
    
    echo "Найдено ${#METRIC_FILES[@]} файлов с метриками"
    
    # Запускаем объединение
    python /tmp/merge_metrics.py "${MERGED_CSV}" "${METRIC_FILES[@]}"
fi

echo "Анализ файла: ${MERGED_CSV}"
echo "Результаты будут сохранены в: ${ANALYSIS_DIR}"

python analyze_metrics.py \
    --csv "${MERGED_CSV}" \
    --output-dir "${ANALYSIS_DIR}"



