import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .config import load_cfg, ensure_dirs
from .data import load_test, make_input_text, swap_row, LABELS
from .model import CrossEncoderClassifier

class TestDataset(Dataset):
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
        return item

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for batch in tqdm(loader, desc="infer"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids, attn)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    ensure_dirs(cfg)

    ckpt_path = Path(cfg.output_dir) / f"fold{cfg.fold}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer"], use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoderClassifier(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    df = load_test(str(Path(cfg.data_dir) / "test.csv"))

    ds = TestDataset(df, tokenizer, cfg.max_length)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2)
    probs = predict_probs(model, dl, device)

    # TTA swap
    if cfg.tta_swap:
        df_sw = df.copy()
        df_sw[["prompt", "response_a", "response_b"]] = df_sw.apply(
            lambda r: swap_row(r["prompt"], r["response_a"], r["response_b"], None)[:3],
            axis=1,
            result_type="expand",
        )
        ds2 = TestDataset(df_sw, tokenizer, cfg.max_length)
        dl2 = DataLoader(ds2, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2)
        probs2 = predict_probs(model, dl2, device)
        probs2 = probs2[:, [1, 0, 2]]
        probs = 0.5 * (probs + probs2)

    sub = pd.DataFrame({
        "id": df["id"].astype(str),
        LABELS[0]: probs[:, 0],
        LABELS[1]: probs[:, 1],
        LABELS[2]: probs[:, 2],
    })
    sub_path = Path(cfg.output_dir) / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote: {sub_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)