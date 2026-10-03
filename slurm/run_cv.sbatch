#!/bin/bash
#SBATCH --job-name=haps_cv
#SBATCH --partition=gpu
#SBATCH --nodes=1                                  # node count
#SBATCH --ntasks-per-node=1                        # total number of tasks per node
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch_logs/log_%x_%j.txt

# Полный nested CV прогон одной/нескольких метрик.
#
# Запуск (позиционные аргументы — безопасно для списков с запятыми):
#   cd slurm
#   sbatch run_cv.sbatch "cellpose,lpips_cellpose" ../results/deep.pkl
#
#   $1 = IDENTITIES (список метрик через запятую)
#   $2 = OUTPUT (необязательно; по умолчанию ../results/<ids>_<jobid>.pkl)
#
# Прочие параметры — через --export=ALL,CSV=...,NJOBS=...,OUTER_SEED=...
# ВНИМАНИЕ: не передавайте списки с запятыми через --export (sbatch трактует запятую как разделитель переменных) — используйте позиционный $1.

export REPO="${REPO:-/beegfs/home/f.gubanov/f.gubanov/bimai_lab/HAPS_v1_Vladislav}"
source "${REPO}/slurm/env.sh"

IDENTITIES="${1:-${IDENTITIES:-ncc,psnr,mi,ssim}}"
OUTPUT="${2:-${OUTPUT:-}}"
CSV="${CSV:-/beegfs/home/f.gubanov/f.gubanov/bimai_lab/filtration_imgs/exp_full.csv}"
NJOBS="${NJOBS:-1}"
OUTER_SEED="${OUTER_SEED:-155}"
SAFE_ID="$(echo "${IDENTITIES}" | tr ',' '-')"
OUTPUT="${OUTPUT:-${REPO}/results/${SAFE_ID}_${SLURM_JOB_ID}.pkl}"

mkdir -p "$(dirname "${OUTPUT}")"
echo "IDENTITIES=${IDENTITIES}"
echo "CSV=${CSV}  NJOBS=${NJOBS}  OUTER_SEED=${OUTER_SEED}"
echo "OUTPUT=${OUTPUT}"

# Не перезапускаем, если артефакт уже есть (чтобы пересчитать — удалите .pkl вручную).
if [ -f "${OUTPUT}" ]; then
  echo "SKIP: output already exists: ${OUTPUT}"
  echo "  Удалите файл вручную, если хотите перезапустить эксперимент."
  exit 0
fi

"${VENV_PY}" -u run_cv_v1.py \
    --identities "${IDENTITIES}" \
    --csv "${CSV}" \
    --output "${OUTPUT}" \
    --outer-seed "${OUTER_SEED}" \
    --n-jobs "${NJOBS}"

echo "CV DONE -> ${OUTPUT}"
