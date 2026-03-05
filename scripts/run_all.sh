#!/usr/bin/env bash
set -euo pipefail

bash scripts/kaggle_download.sh

echo "=== Running TFIDF baseline ==="
bash scripts/run_tfidf.sh

echo "=== Running DeBERTa baseline ==="
bash scripts/run_deberta.sh