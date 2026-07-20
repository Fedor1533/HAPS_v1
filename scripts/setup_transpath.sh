#!/usr/bin/env bash
# Настройка CTransPath (TransPath) для метрики `transpath`.
#
# CTransPath требует:
#   1) код модели (ctran.py) из репозитория TransPath;
#   2) патченный timm 0.5.4 (ConvStem) — из того же репозитория;
#   3) веса ctranspath.pth (~110 MB, Google Drive).
#
# Веса нельзя надёжно скачать автоматически (Google Drive), поэтому шаг 3 —
# ручной либо через gdown.
#
# Использование:
#   bash scripts/setup_transpath.sh [TARGET_DIR]
# По умолчанию TARGET_DIR = ./TransPath
set -euo pipefail

TARGET_DIR="${1:-TransPath}"

if [ ! -d "${TARGET_DIR}/.git" ]; then
  echo ">> Cloning TransPath into ${TARGET_DIR} ..."
  git clone https://github.com/Xiyue-Wang/TransPath.git "${TARGET_DIR}"
else
  echo ">> TransPath repo already present at ${TARGET_DIR}"
fi

echo ">> Installing patched timm 0.5.4 (ConvStem) ..."
if [ -f "${TARGET_DIR}/timm-0.5.4.tar" ]; then
  pip install "${TARGET_DIR}/timm-0.5.4.tar"
else
  echo "!! ${TARGET_DIR}/timm-0.5.4.tar not found."
  echo "   Скачайте патченный timm по инструкции из README TransPath и установите вручную."
fi

WEIGHTS="${TARGET_DIR}/ctranspath.pth"
if [ ! -f "${WEIGHTS}" ]; then
  echo ">> Downloading ctranspath.pth via gdown ..."
  # file id из README репозитория TransPath
  pip install --quiet gdown || true
  gdown --id 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O "${WEIGHTS}" || {
    echo "!! Автозагрузка не удалась. Скачайте ctranspath.pth вручную (Google Drive"
    echo "   ссылка в README https://github.com/Xiyue-Wang/TransPath) и положите в ${WEIGHTS}"
  }
else
  echo ">> Weights already present: ${WEIGHTS}"
fi

echo ""
echo ">> Готово. Экспортируйте пути перед запуском (если не в ./TransPath):"
echo "     export TRANSPATH_ROOT=$(cd "${TARGET_DIR}" && pwd)"
echo "     export TRANSPATH_WEIGHTS=$(cd "${TARGET_DIR}" && pwd)/ctranspath.pth"
