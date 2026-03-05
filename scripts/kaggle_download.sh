#!/usr/bin/env bash
set -euo pipefail

COMP=llm-classification-finetuning
mkdir -p data/raw

kaggle competitions download -c "$COMP" -p data/raw --unzip

echo "Downloaded into data/raw"
ls -lh data/raw