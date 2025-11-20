# src/aiservices/routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
import os

from .config import settings

# ✅ Use package-relative imports (works inside src/aiservices)
from src.services.data_quality.feature_type_inference import detect_feature_types
from src.services.data_quality.data_imputation import impute_missing
from src.services.data_quality.anomaly_detection import anomaly_score
from src.services.data_quality.personalized_detection import apply_rules

# XGBOD artifacts + scorer
from src.services.outlier_detection.xgbod_runtime import load_artifacts, score_xgbod

try:
    # If you created a helper `detect_outliers(df)` in your package, import it:
    from services.outlier_detection.outlier_detection import detect_outliers  # noqa: F401
    HAS_DETECT_OUTLIERS = True
except Exception:
    HAS_DETECT_OUTLIERS = False

api_bp = Blueprint("api", __name__)

# Hard guardrails
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "50000"))   # cap rows
REJECT_EMPTY = True

def _as_df():
    """
    Expect JSON: {"records": [ {...}, {...}, ... ]}
    Returns: (DataFrame, original_body_dict)
    """
    body = request.get_json(force=True, silent=False)
    if not isinstance(body, dict) or "records" not in body:
        raise ValueError("Body must be {'records': [ {...}, ... ]}")
    if not isinstance(body["records"], list):
        raise ValueError("'records' must be a list of objects")
    return pd.DataFrame(body["records"]), body
    n = len(body["records"])
    if REJECT_EMPTY and n == 0:
        raise ValueError("'records' must not be empty")
    if n > MAX_RECORDS:
        raise ValueError(f"'records' too large: {n} > {MAX_RECORDS}")

# -----------------------------
# Data Quality endpoints
# -----------------------------
@api_bp.post("/v1/data-quality/feature-type")
def api_feature_type():
    try:
        df, _ = _as_df()
        return jsonify({"types": detect_feature_types(df)})
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


@api_bp.post("/v1/data-quality/impute")
def api_impute():
    try:
        df, body = _as_df()
        strat = (body.get("strategy") or "mean").lower()
        out = impute_missing(df, strategy=strat)
        return jsonify({"records": out.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


@api_bp.post("/v1/data-quality/anomaly")
def api_anomaly():
    try:
        df, _ = _as_df()
        out = anomaly_score(df)
        return jsonify({"records": out.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


@api_bp.post("/v1/data-quality/personalized")
def api_personalized():
    try:
        df, body = _as_df()
        rules = body.get("rules") or []
        out = apply_rules(df, rules)
        return jsonify({"records": out.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


# -----------------------------
# Outlier endpoints
# -----------------------------
@api_bp.post("/v1/outliers/detect")
def api_outliers_iso():
    """
    Simple outlier detection (IsolationForest).
    If you have a helper `detect_outliers(df)` in your package, it will be used.
    Otherwise we run a safe inline fallback on numeric columns only.
    """
    try:
        df, _ = _as_df()

        if HAS_DETECT_OUTLIERS:
            # Use your packaged implementation
            from src.services.outlier_detection.outlier_detection import detect_outliers
            out = detect_outliers(df)
            return jsonify({"records": out.to_dict(orient="records")})

        # Fallback inline implementation
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        num_df = df.select_dtypes(include=["number"]).copy()
        if num_df.shape[1] == 0:
            out = df.copy()
            out["detected_outliers"] = 0
            return jsonify({"records": out.to_dict(orient="records")})

        X = StandardScaler().fit_transform(num_df.values)
        model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
        scores = model.fit_predict(X)  # -1 anomaly, 1 normal

        out = df.copy()
        out["detected_outliers"] = (scores == -1).astype(int)
        return jsonify({"records": out.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"detail": str(e)}), 400


@api_bp.post("/v1/outliers/xgbod")
def api_outliers_xgbod():
    try:
        df, body = _as_df()
        artifacts = load_artifacts()
        ok, msg, out, thr, info = score_xgbod(
            df, artifacts,
            include_score=bool(body.get("include_score", False)),
            strict_schema=bool(body.get("strict_schema", False)),
            coerce_numeric=bool(body.get("coerce_numeric", True)),
            fill_value=float(body.get("fill_value", 0.0)),
            greater_is_outlier=True
        )
        if not ok:
            return jsonify({"detail": msg}), 400

        return jsonify({
            "records": out.to_dict(orient="records"),
            "threshold": thr,
            "info": info
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 400
