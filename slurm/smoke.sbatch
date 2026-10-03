#!/bin/bash
#SBATCH --job-name=haps_smoke
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Smoke-тест метрик на нескольких парах (быстро, для проверки окружения/кода).
#
# Запуск:
#   cd slurm && sbatch smoke.sbatch
# Переопределение параметров:
#   sbatch --export=ALL,CSV=/path.csv,N=8,METRICS="ncc,cellpose,transpath" smoke.sbatch

export REPO="${REPO:-/trinity/home/vladislav.kozlovskiy/HAPS/HAPS_v1}"
source "${REPO}/slurm/env.sh"

CSV="${CSV:-/trinity/home/vladislav.kozlovskiy/archieve/filtration_imgs/exp_full_local.csv}"
N="${N:-6}"
METRICS="${METRICS:-ncc,psnr,mi,ssim,ms-ssim,fsim,fsimc,lpips_alex,dists,cellpose,lpips_cellpose}"

echo "CSV=${CSV}  N=${N}  METRICS=${METRICS}"
"${VENV_PY}" scripts/smoke_test.py --csv "${CSV}" --n "${N}" --metrics "${METRICS}"
echo "SMOKE DONE"
