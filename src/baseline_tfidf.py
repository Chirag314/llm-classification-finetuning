import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import load_train, load_test, make_input_text, LABELS

def main(data_dir="data/raw", out_dir="outputs/tfidf_lr", n_splits=5, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_dir) / "train.csv"
    test_path = Path(data_dir) / "test.csv"
    df = load_train(str(train_path))
    te = load_test(str(test_path))

    X_text = df.apply(lambda r: make_input_text(r["prompt"], r["response_a"], r["response_b"]), axis=1).tolist()
    y = df["label"].values

    X_test_text = te.apply(lambda r: make_input_text(r["prompt"], r["response_a"], r["response_b"]), axis=1).tolist()

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
    )

    X = vec.fit_transform(X_text)
    X_test = vec.transform(X_test_text)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros((len(df), 3))
    test_pred = np.zeros((len(te), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            multi_class="multinomial",
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])
        test_pred += clf.predict_proba(X_test) / n_splits

        fold_ll = log_loss(y[va_idx], oof[va_idx], labels=[0, 1, 2])
        print(f"fold {fold} logloss: {fold_ll:.6f}")

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
    print(f"wrote {sub_path}")

if __name__ == "__main__":
    main()