import operator
import pandas as pd
from typing import List, Dict, Any

OPS = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt,
    ">=": operator.ge, "<=": operator.le,
    "contains": lambda a, b: a.astype(str).str.contains(str(b), na=False),
}

def apply_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    if not rules:
        out["_personalized_flag"] = False
        return out
    mask = None
    for r in rules:
        col, op, val = r.get("column"), r.get("op"), r.get("value")
        if col not in out.columns or op not in OPS:
            continue
        current = OPS[op](out[col], val) if op != "contains" else OPS[op](out[col], val)
        mask = current if mask is None else (mask | current)
    out["_personalized_flag"] = mask.fillna(False) if mask is not None else False
    return out
