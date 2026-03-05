import ast
import pandas as pd

LABELS = ["winner_model_a", "winner_model_b", "winner_tie"]

def _safe_parse_list_str(x: str) -> str:
    """
    Many rows store prompt/response as a stringified list like:
      "['text here']"
    and may include 'null' in that string.
    We'll parse robustly and return first element as string.
    """
    if not isinstance(x, str):
        return "" if pd.isna(x) else str(x)
    s = x.replace("null", "''")
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list) and len(val) > 0:
            return "" if val[0] is None else str(val[0])
        return str(val)
    except Exception:
        return x

def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].map(_safe_parse_list_str)

    df["label_name"] = df[LABELS].idxmax(axis=1)
    df["label"] = df["label_name"].map({LABELS[0]: 0, LABELS[1]: 1, LABELS[2]: 2}).astype(int)
    return df

def load_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].map(_safe_parse_list_str)
    return df

def make_input_text(prompt: str, ra: str, rb: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse A:\n{ra}\n\nResponse B:\n{rb}"

def swap_row(prompt: str, ra: str, rb: str, label: int | None):
    new_label = None
    if label is not None:
        if label == 0:
            new_label = 1
        elif label == 1:
            new_label = 0
        else:
            new_label = 2
    return prompt, rb, ra, new_label