#!/bin/bash
# Общие настройки окружения для slurm-задач HAPS_v1.
# Использует uv-окружение (.venv) напрямую — БЕЗ обращения к сети на compute-ноде.
set -euo pipefail

# Корень репозитория (можно переопределить переменной REPO).
export REPO="${REPO:-/trinity/home/vladislav.kozlovskiy/HAPS/HAPS_v1}"

# Python из uv-venv. Если venv нет — подсказка про uv sync (делать на login-ноде с сетью).
export VENV_PY="${REPO}/.venv/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: ${VENV_PY} не найден. На login-ноде выполните: cd ${REPO} && uv sync" >&2
  exit 1
fi

# Кэш весов моделей (cellpose / torch hub) — в HOME, чтобы был доступен на compute-нодах.
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-$HOME/.cellpose/models}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# TransPath (метрика transpath) — опционально. Раскомментируйте и задайте пути:
# export TRANSPATH_ROOT="${REPO}/TransPath"
# export TRANSPATH_WEIGHTS="${REPO}/TransPath/ctranspath.pth"

cd "${REPO}"
echo "REPO=${REPO}"
echo "PYTHON=${VENV_PY}"
"${VENV_PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
