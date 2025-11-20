# src/services/outlier_detection/xgbod_runtime.py
from pathlib import Path
from joblib import load as joblib_load
import os, json, pandas as pd, numpy as np

ARTIFACTS_DIR = Path(os.getenv("XGBOD_ARTIFACTS_DIR", "artifacts")).resolve()

MODEL_FILES     = ["xgbod_detector.joblib"]
SCALER_FILES    = ["scaler_XGBOD.joblib"]
THRESH_FILES    = ["xgbod_threshold.joblib", "threshold.joblib"]
FEATURES_FILES  = ["feature_columns.json", "features.json"]

def _find(cands):
    for n in cands:
        p = ARTIFACTS_DIR / n
        if p.exists(): return p
    for n in cands:
        p = Path(n)
        if p.exists(): return p
    return None

def load_artifacts():
    a = {"model":None, "scaler":None, "threshold":None, "features":None, "messages":[]}

    p = _find(MODEL_FILES)
    if p:
        try:
            a["model"] = joblib_load(p)
            a["messages"].append(f"Model loaded: {p.name}")
        except Exception as e:
            a["messages"].append(f"Model load failed ({p.name}): {e}")
    else:
        a["messages"].append("Model file not found.")

    p = _find(SCALER_FILES)
    if p:
        try:
            a["scaler"] = joblib_load(p)
            a["messages"].append(f"Scaler loaded: {p.name}")
        except Exception as e:
            a["messages"].append(f"Scaler load failed ({p.name}): {e}")
    else:
        a["messages"].append("Scaler file not found.")

    p = _find(THRESH_FILES)
    if p:
        try:
            obj = joblib_load(p)
            a["threshold"] = float(obj["threshold"]) if isinstance(obj, dict) and "threshold" in obj else float(obj)
            a["messages"].append(f"Threshold loaded: {p.name} (value: {a['threshold']})")
        except Exception as e:
            a["messages"].append(f"Threshold load failed ({p.name}): {e}")
    else:
        a["messages"].append("Threshold file not found.")

    p = _find(FEATURES_FILES)
    if p:
        try:
            spec = json.loads(Path(p).read_text(encoding="utf-8"))
            feats = spec["features"] if isinstance(spec, dict) and "features" in spec else spec
            if not isinstance(feats, list):
                raise ValueError("features file must be a list or {'features': [...]} .")
            a["features"] = feats
            a["messages"].append(f"Features loaded: {p.name} ({len(feats)} columns)")
        except Exception as e:
            a["messages"].append(f"Features load failed ({p.name}): {e}")
    else:
        a["messages"].append("Features file not found.")

    return a

def score_xgbod(df, artifacts, *, include_score=False, strict_schema=False,
                coerce_numeric=True, fill_value=0.0, greater_is_outlier=True):
    # readiness
    missing_keys = [k for k in ("model","scaler","threshold","features") if artifacts.get(k) is None]
    if missing_keys:
        return (False,
                "Model artifacts not ready/missing: " + ", ".join(missing_keys) +
                ". " + " | ".join(artifacts.get("messages", [])),
                None, None, None)

    feats, scaler, model, thr = artifacts["features"], artifacts["scaler"], artifacts["model"], artifacts["threshold"]

    extra = [c for c in df.columns if c not in feats]
    missing = [c for c in feats if c not in df.columns]
    if strict_schema and missing:
        return False, f"Missing required feature columns: {missing}", None, None, None

    Xdf = df.reindex(columns=feats)
    if coerce_numeric:
        for c in feats:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.fillna(fill_value)

    try:
        Xs = scaler.transform(Xdf.values)
    except Exception as e:
        return False, f"Scaler transform failed: {e}", None, None, None

    try:
        scores = model.decision_function(Xs)
    except Exception as e:
        return False, f"Scoring failed: {e}", None, None, None

    labels = (scores > thr).astype(int) if greater_is_outlier else (scores < thr).astype(int)

    out = df.copy()
    if include_score:
        out["od_score"] = scores
    out["detected outliers"] = labels

    info = []
    if not strict_schema:
        if missing: info.append(f"Missing features were added and filled with {fill_value}: {missing}")
        if extra:   info.append(f"Ignored extra columns (kept in output): {extra[:10]}{'...' if len(extra)>10 else ''}")

    return True, "", out, thr, info
