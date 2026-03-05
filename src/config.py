from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class CFG:
    seed: int
    data_dir: str
    output_dir: str
    model_name: str
    max_length: int

    num_folds: int
    fold: int
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    label_smoothing: float

    use_swap_augmentation: bool
    tta_swap: bool

def load_cfg(path: str) -> CFG:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return CFG(**d)

def ensure_dirs(cfg: CFG) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out