#!/bin/bash
#SBATCH --job-name=haps_cv_cpu
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch_logs/log_%x_%j.out

# Nested CV для CPU-only метрик (структурные: ngf / tile_* / flow_residual,
# а также классические ncc/psnr/mi/ssim). GPU не запрашивается, поэтому джоб
# идёт в htc и не занимает лимит GPU-очереди.
#
#   $1 = IDENTITIES (список метрик через запятую)
#   $2 = OUTPUT
#
# Outer-фолды считаются параллельно (NJOBS), поэтому BLAS/OpenCV прижаты к
# одному потоку на процесс — иначе 5 процессов по 8 потоков передерутся за ядра.

export REPO="${REPO:-/trinity/home/vladislav.kozlovskiy/HAPS/HAPS_v1}"
source "${REPO}/slurm/env.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

IDENTITIES="${1:-${IDENTITIES:-ngf}}"
OUTPUT="${2:-${OUTPUT:-}}"
CSV="${CSV:-/trinity/home/vladislav.kozlovskiy/archieve/filtration_imgs/exp_full_local.csv}"
NJOBS="${NJOBS:-5}"
OUTER_SEED="${OUTER_SEED:-155}"
SAFE_ID="$(echo "${IDENTITIES}" | tr ',' '-')"
OUTPUT="${OUTPUT:-${REPO}/results/${SAFE_ID}_${SLURM_JOB_ID}.pkl}"

mkdir -p "$(dirname "${OUTPUT}")"
echo "IDENTITIES=${IDENTITIES}"
echo "CSV=${CSV}  NJOBS=${NJOBS}  OUTER_SEED=${OUTER_SEED}"
echo "OUTPUT=${OUTPUT}"

if [ -f "${OUTPUT}" ]; then
  echo "SKIP: output already exists: ${OUTPUT}"
  exit 0
fi

"${VENV_PY}" -u run_cv_v1.py \
    --identities "${IDENTITIES}" \
    --csv "${CSV}" \
    --output "${OUTPUT}" \
    --outer-seed "${OUTER_SEED}" \
    --n-jobs "${NJOBS}"

echo "CV DONE -> ${OUTPUT}"
