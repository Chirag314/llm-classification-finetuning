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

#python -m src.submit --config "$CFG" --message "deberta-v3-base + swap aug + swap TTA"
#!/usr/bin/env bash
set -euo pipefail

CFG=configs/deberta_base.yaml

python -m src.train --config "$CFG"
python -m src.infer --config "$CFG"

python - <<'PY'
from src.plotting import plot_history
plot_history("outputs/deberta_base/train_history.json", "outputs/deberta_base/train_curve.png")
PY

echo "Created: outputs/deberta_base/submission.csv"
echo "This competition is notebook-only, so submit from a Kaggle Notebook."