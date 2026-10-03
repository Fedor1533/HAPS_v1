#!/usr/bin/env bash
# Сабмит nested-CV джобов для всех HF foundation-метрик (по одному на модель).
# Партиция: только gpu (одна).
#
# Usage:
#   bash slurm/submit_foundation.sh
#   MODELS="phikon hibou_l" bash slurm/submit_foundation.sh
#   SUFFIX=_spec bash slurm/submit_foundation.sh    # spec-точный препроцессинг, rgb-only
set -euo pipefail

REPO="${REPO:-/trinity/home/vladislav.kozlovskiy/HAPS/HAPS_v1}"
PARTITION="${PARTITION:-gpu}"
MODELS="${MODELS:-phikon hibou_l uni2h gigapath virchow2 hoptimus0 ctranspath_hf cpsam}"
# Отдельный venv с modern timm — не трогает .venv (patched timm-0.5.4 для local transpath).
VENV_PY="${VENV_PY:-${REPO}/.venv-foundation/bin/python}"

if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: ${VENV_PY} не найден. Сначала: bash scripts/setup_foundation.sh" >&2
  exit 1
fi

cd "${REPO}/slurm"
mkdir -p "${REPO}/results"

SUFFIX="${SUFFIX:-}"

for m in ${MODELS}; do
  ident="${m}${SUFFIX}"
  out="${REPO}/results/${ident}.pkl"
  if [ -f "${out}" ]; then
    echo "SKIP ${ident}: already exists ${out}"
    continue
  fi
  job=$(sbatch --parsable --partition="${PARTITION}" \
        --export=ALL,REPO="${REPO}",VENV_PY="${VENV_PY}" \
        run_cv.sbatch "${ident}" "${out}")
  echo "SUBMITTED ${ident} -> job ${job} -> ${out}  (venv=${VENV_PY})"
done

echo "---- queue ----"
squeue -u "$USER" -o "%.10i %.12j %.10P %.2t %.10M %.20R"
