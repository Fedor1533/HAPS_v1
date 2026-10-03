#!/bin/bash
# Общие настройки окружения для slurm-задач HAPS_v1.
# Использует uv-окружение (.venv) напрямую — БЕЗ обращения к сети на compute-ноде.
set -euo pipefail

# Корень репозитория (можно переопределить переменной REPO).
export REPO="${REPO:-/beegfs/home/f.gubanov/f.gubanov/bimai_lab/HAPS_v1_Vladislav}"

# Python из uv-venv. Можно переопределить VENV_PY (например .venv-foundation для HF timm).
# Если venv нет — подсказка про uv sync (делать на login-ноде с сетью).
export VENV_PY="${VENV_PY:-${REPO}/.venv/bin/python}"
if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: ${VENV_PY} не найден. На login-ноде выполните: cd ${REPO} && uv sync" >&2
  exit 1
fi

# Кэш весов моделей (cellpose / torch hub / HF) — в HOME, доступен на compute-нодах.
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-$HOME/.cellpose/models}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Общий кэш HF-моделей лаборатории (foundation-бэкбоны уже скачаны коллегой).
# По умолчанию читаем offline из этого кэша. Для докачки новых gated-моделей
# переопределите HF_HUB_CACHE (напр. $HF_HOME/hub) и снимите HF_HUB_OFFLINE.
_LAB_HF_CACHE="/trinity/home/vladislav.kozlovskiy/.cache/huggingface/hub"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${_LAB_HF_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${_LAB_HF_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${_LAB_HF_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Токен HF: из env либо из файла login'а (не хранить в репо).
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

# TransPath (метрика transpath)
export TRANSPATH_ROOT="${TRANSPATH_ROOT:-${REPO}/TransPath}"
export TRANSPATH_WEIGHTS="${TRANSPATH_WEIGHTS:-${REPO}/TransPath/ctranspath.pth}"

cd "${REPO}"
echo "REPO=${REPO}"
echo "PYTHON=${VENV_PY}"
"${VENV_PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
