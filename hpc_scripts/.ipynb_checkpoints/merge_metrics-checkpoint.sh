#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=htc
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=../logs/merge_metrics.out
#SBATCH --error=../logs/merge_metrics.err

# Активация окружения
unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain

# Переходим в рабочую директорию
cd ~/benchmarking/HAPS_v1

set -e

# =============================================
# Конфигурация
# =============================================

RESULTS_DIR="${HOME}/benchmarking/HAPS_v1/results"
MERGED_CSV="${RESULTS_DIR}/all_metrics_merged_test.csv"

echo "============================================="
echo " ОБЪЕДИНЕНИЕ МЕТРИК"
echo "============================================="
echo "Директория с результатами: ${RESULTS_DIR}"
echo "Выходной файл: ${MERGED_CSV}"
echo "============================================="

# Проверяем наличие файлов
echo ""
echo "Проверка наличия файлов:"
for METRIC in ncc lpips; do
    for SPATIAL in full downsample patches; do
        FILE="${RESULTS_DIR}/mist_${METRIC}_${SPATIAL}_test.csv"
        if [ -f "${FILE}" ]; then
            echo "  ✓ $(basename ${FILE})"
        else
            echo "  ✗ Отсутствует: $(basename ${FILE})"
        fi
    done
done

# Запускаем объединение
echo ""
echo "Запуск объединения..."
python merge_metrics.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${MERGED_CSV}"

echo ""
echo "============================================="
echo " ГОТОВО!"
echo "============================================="
echo "Объединенный файл: ${MERGED_CSV}"
echo ""
echo "Для анализа выполните:"
echo "  python analyze_metrics.py --csv ${MERGED_CSV} --output-dir ${HOME}/benchmarking/HAPS_v1/analysis"