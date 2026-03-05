import subprocess
from pathlib import Path
from .config import load_cfg

def main(cfg_path: str, message: str):
    cfg = load_cfg(cfg_path)
    sub_path = Path(cfg.output_dir) / "submission.csv"
    if not sub_path.exists():
        raise FileNotFoundError(f"Missing {sub_path}. Run infer first.")
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "llm-classification-finetuning",
        "-f", str(sub_path),
        "-m", message,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--message", default="baseline submission")
    args = ap.parse_args()
    main(args.config, args.message)