#!/usr/bin/env bash
# Ставит отдельный .venv-foundation с modern timm для HF foundation-метрик
# (UNI2/Virchow2/H-optimus/GigaPath/ctranspath_hf). Не трогает .venv
# (patched timm-0.5.4 для локального `transpath`).
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO}"
export PATH="$HOME/.local/bin:$PATH"

VENV_DIR="${REPO}/.venv-foundation"
# /tmp на login часто забит (root ~100%) — кэш uv держим на beegfs.
mkdir -p "${REPO}/.uv-tmp" "${REPO}/.uv-cache"
export TMPDIR="${TMPDIR:-${REPO}/.uv-tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO}/.uv-cache}"
echo ">> UV_PROJECT_ENVIRONMENT=${VENV_DIR} uv sync --extra foundation"
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --extra foundation

# Cellpose-SAM (cpsam): cellpose>=4 + segment-anything. Не пишем в pyproject,
# чтобы основной .venv остался на cellpose 2.x (cyto2 ResUNet).
echo ">> upgrade cellpose>=4 in .venv-foundation (for cpsam)"
uv pip install --python "${VENV_DIR}/bin/python" \
  'cellpose>=4.2,<5' 'segment-anything==1.0' \
  'numpy>=1.26,<2' 'opencv-python-headless>=4.8,<4.11'

echo ">> timm / cellpose (.venv-foundation):"
"${VENV_DIR}/bin/python" -c "import timm, cellpose; from importlib.metadata import version; print('timm', timm.__version__, 'cellpose', version('cellpose'))"
echo ">> timm / cellpose (.venv, local transpath / cyto2):"
.venv/bin/python -c "import timm; from importlib.metadata import version; print('timm', timm.__version__, 'cellpose', version('cellpose'))" 2>/dev/null || true

echo ""
echo ">> Токен HF: ~/.cache/huggingface/token (или HF_TOKEN)."
echo ">> Сабмит: bash slurm/submit_foundation.sh"
