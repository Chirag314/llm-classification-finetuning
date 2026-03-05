from pathlib import Path
import json
import matplotlib.pyplot as plt

def plot_history(history_path: str, out_png: str):
    p = Path(history_path)
    if not p.exists():
        print(f"[plot] missing history json: {history_path}")
        return

    hist = json.loads(p.read_text(encoding="utf-8"))
    steps = [h["step"] for h in hist]
    tr = [h["train_loss"] for h in hist]
    va = [h["valid_logloss"] for h in hist]

    plt.figure()
    plt.plot(steps, tr, label="train_loss")
    plt.plot(steps, va, label="valid_logloss")
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] wrote {out_png}")