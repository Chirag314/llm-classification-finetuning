import ast
import json
import pandas as pd

LABELS = ["winner_model_a", "winner_model_b", "winner_tie"]

def _safe_parse_list_str(x):
    """
    Competition text columns are often stored like:
      "['some text']"
    or JSON-like strings.
    Return the first element if it's a list, else return string as-is.
    """
    if not isinstance(x, str):
        return "" if pd.isna(x) else str(x)

    s = x.strip()
    if not s:
        return ""

    # Try JSON first
    try:
        val = json.loads(s)
        if isinstance(val, list) and len(val) > 0:
            return "" if val[0] is None else str(val[0])
        return str(val)
    except Exception:
        pass

    # Try Python literal safely
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list) and len(val) > 0:
            return "" if val[0] is None else str(val[0])
        return str(val)
    except Exception:
        return s

def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].map(_safe_parse_list_str)

    df["label_name"] = df[LABELS].idxmax(axis=1)
    df["label"] = df["label_name"].map({
        LABELS[0]: 0,
        LABELS[1]: 1,
        LABELS[2]: 2
    }).astype(int)
    return df

def load_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].map(_safe_parse_list_str)
    return df

def _truncate_text(s: str, max_chars: int) -> str:
    s = "" if s is None else str(s)
    return s[:max_chars]

def make_input_text(prompt: str, ra: str, rb: str, max_prompt_chars: int = 600, max_resp_chars: int = 1000) -> str:
    prompt = _truncate_text(prompt, max_prompt_chars)
    ra = _truncate_text(ra, max_resp_chars)
    rb = _truncate_text(rb, max_resp_chars)

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