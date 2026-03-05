#!/usr/bin/env bash
set -euo pipefail

python -m src.baseline_tfidf
kaggle competitions submit -c llm-classification-finetuning -f outputs/tfidf_lr/submission.csv -m "TFIDF+LogReg baseline"