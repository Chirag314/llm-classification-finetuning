import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from transformers import DebertaV2Tokenizer, get_cosine_schedule_with_warmup

from .config import load_cfg, ensure_dirs
from .data import load_train, make_input_text, swap_row, LABELS
from .model import CrossEncoderClassifier

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PrefDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = make_input_text(r["prompt"], r["response_a"], r["response_b"])
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["label"] = torch.tensor(int(r["label"]), dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()
        logits = model(input_ids, attn)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return log_loss(all_labels, all_probs, labels=[0, 1, 2])

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed_everything(cfg.seed)
    outdir = ensure_dirs(cfg)

    df = load_train(str(Path(cfg.data_dir) / "train.csv"))

    # swap augmentation
    if cfg.use_swap_augmentation:
        aug = []
        for _, r in df.iterrows():
            p, ra, rb, y = swap_row(r["prompt"], r["response_a"], r["response_b"], int(r["label"]))
            aug.append({"prompt": p, "response_a": ra, "response_b": rb, "label": y})
        df_aug = pd.concat(
            [df[["prompt", "response_a", "response_b", "label"]], pd.DataFrame(aug)],
            ignore_index=True,
        )
    else:
        df_aug = df[["prompt", "response_a", "response_b", "label"]].copy()

    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
    folds = list(skf.split(df_aug, df_aug["label"]))
    tr_idx, va_idx = folds[cfg.fold]
    df_tr = df_aug.iloc[tr_idx].reset_index(drop=True)
    df_va = df_aug.iloc[va_idx].reset_index(drop=True)

    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.model_name)
    ds_tr = PrefDataset(df_tr, tokenizer, cfg.max_length)
    ds_va = PrefDataset(df_va, tokenizer, cfg.max_length)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoderClassifier(cfg.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = max(1, (len(dl_tr) * cfg.epochs) // cfg.grad_accum_steps)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best = 1e9
    history = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{cfg.epochs}")
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        count = 0

        for it, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids, attn)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.0,
                )
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * cfg.grad_accum_steps
            count += 1

            if (it + 1) % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                avg_train_loss = running / max(1, count)
                running, count = 0.0, 0

                pbar.set_postfix(train_loss=avg_train_loss)

                # periodic validation
                if global_step % 50 == 0:
                    va_ll = evaluate(model, dl_va, device)
                    history.append({
                        "step": global_step,
                        "train_loss": avg_train_loss,
                        "valid_logloss": va_ll,
                    })
                    print(f"\nstep {global_step} valid logloss: {va_ll:.6f}")

                    if va_ll < best:
                        best = va_ll
                        torch.save(
                            {
                                "model_name": cfg.model_name,
                                "state_dict": model.state_dict(),
                                "tokenizer": cfg.model_name,
                                "cfg": cfg.__dict__,
                                "best_logloss": best,
                            },
                            Path(cfg.output_dir) / f"fold{cfg.fold}_best.pt",
                        )
                        print("saved best checkpoint")

        # end-of-epoch validation
        va_ll = evaluate(model, dl_va, device)
        history.append({"step": global_step, "train_loss": None, "valid_logloss": va_ll})
        print(f"epoch end valid logloss: {va_ll:.6f}")

        if va_ll < best:
            best = va_ll
            torch.save(
                {
                    "model_name": cfg.model_name,
                    "state_dict": model.state_dict(),
                    "tokenizer": cfg.model_name,
                    "cfg": cfg.__dict__,
                    "best_logloss": best,
                },
                Path(cfg.output_dir) / f"fold{cfg.fold}_best.pt",
            )
            print("saved best checkpoint")

    # write history
    hist_path = Path(cfg.output_dir) / "train_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"wrote {hist_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)