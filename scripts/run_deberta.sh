#!/usr/bin/env bash
set -euo pipefail

CFG=configs/deberta_base.yaml

python -m src.train --config "$CFG"
python -m src.infer --config "$CFG"

# plot
python - <<'PY'
from src.plotting import plot_history
plot_history("outputs/deberta_base/train_history.json", "outputs/deberta_base/train_curve.png")
PY

python -m src.submit --config "$CFG" --message "deberta-v3-base + swap aug + swap TTA"