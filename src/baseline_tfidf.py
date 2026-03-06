import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import load_train, load_test, make_input_text, LABELS

def main(data_dir="data/raw", out_dir="outputs/tfidf_lr", n_splits=3, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_dir) / "train.csv"
    test_path = Path(data_dir) / "test.csv"

    print("Loading data...")
    df = load_train(str(train_path))
    te = load_test(str(test_path))

    print("Building text fields...")
    X_text = df.apply(
        lambda r: make_input_text(
            r["prompt"],
            r["response_a"],
            r["response_b"],
            max_prompt_chars=400,
            max_resp_chars=700,
        ),
        axis=1,
    ).tolist()

    y = df["label"].values

    X_test_text = te.apply(
        lambda r: make_input_text(
            r["prompt"],
            r["response_a"],
            r["response_b"],
            max_prompt_chars=400,
            max_resp_chars=700,
        ),
        axis=1,
    ).tolist()

    print("Vectorizing text...")
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )

    X = vec.fit_transform(X_text)
    X_test = vec.transform(X_test_text)

    print("Training CV model...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros((len(df), 3))
    test_pred = np.zeros((len(te), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")

        clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="saga",
            C=3.0,
            n_jobs=-1,
            verbose=0,
        )

        clf.fit(X[tr_idx], y[tr_idx])

        val_pred = clf.predict_proba(X[va_idx])
        oof[va_idx] = val_pred
        test_pred += clf.predict_proba(X_test) / n_splits

        fold_ll = log_loss(y[va_idx], val_pred, labels=[0, 1, 2])
        print(f"  fold logloss: {fold_ll:.6f}")

    overall = log_loss(y, oof, labels=[0, 1, 2])
    print(f"OOF logloss: {overall:.6f}")

    sub = pd.DataFrame({
        "id": te["id"].astype(str),
        LABELS[0]: test_pred[:, 0],
        LABELS[1]: test_pred[:, 1],
        LABELS[2]: test_pred[:, 2],
    })

    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}")

if __name__ == "__main__":
    main()