# 🏆 LLM Classification Finetuning — Kaggle Preference Modeling Pipeline

A production-minded, end-to-end ML repository for the Kaggle **LLM Classification Finetuning** competition.

Given a **prompt** and two **anonymous model responses** (A/B), we predict the human preference:

✅ **winner_model_a**  
✅ **winner_model_b**  
✅ **winner_tie**

This repo is built to feel like a ML team codebase:

- Reproducible experiment configs (YAML)
- Baseline → strong Transformer training pipeline
- Probability-calibrated outputs for **LogLoss**
- Fully runnable in **GitHub Codespaces**
- Kaggle API: **download data + submit** from terminal
- Training curve plots saved as artifacts and embedded in README
- Easy path to: **train big on desktop → deploy/infer on Kaggle**

---

## 🔖 Badges

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" />
  <img src="https://img.shields.io/badge/kaggle-competition%20pipeline-20BEFF" />
  <img src="https://img.shields.io/badge/task-preference%20classification-orange" />
  <img src="https://img.shields.io/badge/metric-multiclass%20logloss-red" />
  <img src="https://img.shields.io/badge/runtime-codespaces%20%7C%20kaggle-green" />
</p>

---

# 🎯 What This Competition Is

You are training a model to act like a **judge**.

Input per row:

- `prompt`
- `response_a`
- `response_b`

Target label (3-class):

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

**Goal:** output **well-calibrated probabilities** for the 3 classes.  
**Metric:** multiclass **LogLoss** (lower is better).  
That means “confidence” matters — not just accuracy.

---

# 🧠 Solution Strategy (High-Leverage)

This repo implements proven preference-modeling patterns:

✅ **Cross-encoder** format (Prompt + A + B → classifier)  
✅ **Swap augmentation** (A↔B, flip label) to reduce position bias  
✅ **Swap TTA** at inference (average original + swapped predictions)  
✅ Clean, reproducible scripts for Kaggle workflow

---

# 📦 Project Structure

```text
llm-classification-finetuning/
│
├── configs/
│   └── deberta_base.yaml          # Experiment config (model, max_len, lr, etc.)
│
├── src/
│   ├── data.py                    # Robust parsing + text formatting
│   ├── baseline_tfidf.py          # Baseline 1: TF-IDF + Logistic Regression
│   ├── model.py                   # Cross-encoder transformer head
│   ├── train.py                   # Train + save best checkpoint
│   ├── infer.py                   # Inference + optional swap-TTA
│   ├── submit.py                  # Kaggle submission helper
│   └── plotting.py                # Training curve plot
│
├── scripts/
│   ├── kaggle_download.sh         # Download + unzip Kaggle data
│   ├── run_tfidf.sh               # Run TF-IDF baseline + submit
│   ├── run_deberta.sh             # Train DeBERTa baseline + plot + submit
│   └── run_all.sh                 # Full pipeline
│
├── data/raw/                      # Kaggle CSVs (downloaded here)
└── outputs/                       # Artifacts (models, plots, submissions)

---

# ⚙️ Setup (GitHub Codespaces)
1) Install dependencies

pip install -r requirements.txt

2) Add Kaggle credentials
Option A (recommended): kaggle.json

Kaggle → Account → Create New API Token

Place kaggle.json in Codespace and run:

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

Verify:

kaggle --version
kaggle competitions list | head
Option B: Environment variables (Codespace secrets)

Set KAGGLE_USERNAME and KAGGLE_KEY as Codespace Secrets and restart the Codespace.

🚀 Quick Start (Download → Train → Submit)
Download data
bash scripts/kaggle_download.sh
Baseline 1: TF-IDF + Logistic Regression (fast sanity baseline)

Creates: outputs/tfidf_lr/submission.csv and submits.

bash scripts/run_tfidf.sh
Baseline 2: DeBERTa Cross-Encoder (strong baseline)

Creates:

outputs/deberta_base/fold0_best.pt

outputs/deberta_base/submission.csv

outputs/deberta_base/train_curve.png (training curve)

bash scripts/run_deberta.sh
Full pipeline (both baselines)
bash scripts/run_all.sh
📈 Training Curve

After running the DeBERTa baseline, the training curve will be generated at:

outputs/deberta_base/train_curve.png

GitHub will render it after you push:

🧪 Baselines Included
✅ Baseline 1 (Fast): TF-IDF + Logistic Regression

Purpose: validate parsing, scoring format, and submission logic quickly.

✅ Baseline 2 (Strong): Transformer Cross-Encoder (DeBERTa)

Purpose: strong competitive baseline with:

swap augmentation

swap test-time augmentation (TTA)

logloss-focused training

🧠 How We Improve From Here (Elite Kaggle Iteration)

Once the pipeline runs end-to-end, typical high-signal improvements:

K-fold ensemble (0..4 folds)

Long-context handling (head+tail truncation per response)

Comparator architectures (reduce A/B position bias further)

Better formatting (separator tokens, normalization)

Hyperparameter tuning for logloss (max_len, smoothing, warmup, LR)

Model upgrades (DeBERTa-large / RoBERTa-large) where compute allows

💻 Strong Models: Desktop Training → Kaggle Inference

If you have a powerful desktop GPU, the best workflow is:

✅ Train heavier models locally (longer max_len, larger backbone, more epochs)
✅ Save checkpoint (.pt or HuggingFace folder)
✅ Upload to Kaggle as a Dataset (or Kaggle Model)
✅ Run Kaggle Notebook inference using attached files under /kaggle/input/...
✅ Submit from Kaggle or from Codespaces

This keeps Kaggle runtime within the 8-hour limits while still using big models.

⚠ Constraints & Notes

Kaggle text fields may be stored as stringified lists — parsing is handled in src/data.py.

This repository targets probability quality (LogLoss), not accuracy.

Codespaces is often CPU-only; Transformer training is best on Kaggle GPU or your desktop GPU.

👤 Author

Chirag Desai
Building reproducible ML systems, Kaggle-ready pipelines, and production-style ML engineering workflows.

⭐ If this repository helps you, consider starring it.

If you want, I can also generate a matching **docs/** section (like your architecture diagram style) for:
- “data flow diagram”
- “training/inference lifecycle”
- “K-fold ensemble plan”
but the above already matches your “single-cell beautiful README” style exactly.
