#!/usr/bin/env bash
set -euo pipefail

COMP=llm-classification-finetuning

# Ensure Kaggle creds exist (prefer kaggle.json)
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    echo "ERROR: Kaggle credentials not found."
    echo "Set KAGGLE_USERNAME and KAGGLE_KEY (Codespace secrets) or create ~/.kaggle/kaggle.json"
    exit 1
  fi
  mkdir -p "$HOME/.kaggle"
  cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
  chmod 600 "$HOME/.kaggle/kaggle.json"
fi

mkdir -p data/raw
cd data/raw

echo "Downloading Kaggle competition data..."
kaggle competitions download -c "$COMP"

echo "Unzipping files..."
unzip -o "*.zip"
rm -f *.zip

echo "Data ready in data/raw"
ls -lh