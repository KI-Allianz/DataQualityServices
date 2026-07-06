# bp.py — HTML UI blueprint (no Keycloak)
from flask import Blueprint, render_template, request, redirect, url_for, session, send_file, jsonify
from pathlib import Path
from io import BytesIO
import os, io, uuid
import pandas as pd
import numpy as np

from zipfile import ZipFile, ZIP_DEFLATED
from src.services.piveau_publish import publish_result
from urllib.parse import urlparse


from ..config import settings
from ..security import sanitize_for_spreadsheet_export


from src.services.outlier_detection.xgbod_runtime import load_artifacts, score_xgbod

ui_bp = Blueprint("ui", __name__, template_folder="templates", static_folder="static",static_url_path="/ui-static")

# -------------------------------------------------------------------
# Health + Home
# -------------------------------------------------------------------
@ui_bp.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@ui_bp.get("/")
def index():
    return render_template(
        "index.html",
        #labelling_url=settings.LABELLING_URL,
        labelling_url=settings.LABELLING_SERVICES_URL,
        #deblurring_url=settings.DEBLURRING_URL,
        )


# -------------------------------------------------------------------
# Helpers / State
# -------------------------------------------------------------------
OUTPUT_DIR = Path(settings.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _as_bool(v):
    """Parse checkbox/select truthy values reliably."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    return v in ("1", "true", "on", "yes", "y")

# -------------------------------------------------------------------
# Datasets & Catalogues  UI
# -------------------------------------------------------------------
@ui_bp.route("/goto/datasets")
def goto_datasets():
    return redirect(settings.datasets_url, code=302)

@ui_bp.route("/goto/catalogues")
def goto_catalogues():
    return redirect(settings.catalogues_url, code=302)

# -------------------------------------------------------------------
# Data Quality UI
# -------------------------------------------------------------------

@ui_bp.get("/services/data-quality")
def data_quality_redirect():
    target = settings.DATAQUALITY_SERVICE_URL
    return redirect(target)

# -------------------------------------------------------------------
# Outlier Detection UI (XGBOD)
# -------------------------------------------------------------------
def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    safe_df = sanitize_for_spreadsheet_export(df)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        safe_df.to_excel(w, index=False, sheet_name="results")
    bio.seek(0)
    return bio.getvalue()

@ui_bp.get("/services/outlier")
def outlier_page():
    a = load_artifacts()
    ready = all([a.get("model"), a.get("scaler"),
                 a.get("threshold") is not None, a.get("features")])
    return render_template("services/outlier.html",
                           ready=ready, artifacts=a, step="upload")

@ui_bp.post("/services/outlier/process")
def outlier_process():
    a = load_artifacts()
    ready = all([a.get("model"), a.get("scaler"),
                 a.get("threshold") is not None, a.get("features")])
    if not ready:
        return render_template("services/outlier.html", artifacts=a, ready=False, step="upload",
                               error="Model artifacts are missing under ./artifacts")

    up = request.files.get("data_file")
    sheet_name = (request.form.get("sheet_name") or "").strip()
    strict_schema  = _as_bool(request.form.get("strict_schema"))
    coerce_numeric = _as_bool(request.form.get("coerce_numeric"))
    include_score  = _as_bool(request.form.get("include_score"))
    try:
        fill_value = float((request.form.get("fill_value") or "0").strip())
    except Exception:
        fill_value = 0.0

    if not up or not up.filename:
        return render_template("services/outlier.html", artifacts=a, ready=True, step="upload",
                               error="Please upload a CSV/XLSX file.")

    # read input
    try:
        if up.filename.lower().endswith((".xlsx", ".xls")):
            bio = io.BytesIO(up.read())
            raw_df = pd.read_excel(bio, sheet_name=sheet_name or 0)
        else:
            raw_df = pd.read_csv(up)
    except Exception as e:
        return render_template("services/outlier.html", artifacts=a, ready=True, step="upload",
                               error=f"Could not read file: {e}")

    ok, msg, res_df, thr, info_msgs = score_xgbod(
        raw_df, a,
        include_score=include_score,
        strict_schema=strict_schema,
        coerce_numeric=coerce_numeric,
        fill_value=fill_value,
        greater_is_outlier=True
    )
    if not ok:
        return render_template("services/outlier.html", artifacts=a, ready=True, step="upload", error=msg)

    run_id = uuid.uuid4().hex
    run_dir = OUTPUT_DIR / f"xgbod_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    res_csv  = run_dir / "results.csv"
    res_xlsx = run_dir / "results.xlsx"
    sanitize_for_spreadsheet_export(res_df).to_csv(res_csv, index=False)
    res_xlsx.write_bytes(_to_excel_bytes(res_df))

    inliers  = res_df.loc[res_df["detected outliers"] == 0]
    outliers = res_df.loc[res_df["detected outliers"] == 1]
    outliers_preview_html = outliers.head(50).to_html(index=False, classes="table table-striped", border=0)
    inliers_preview_html  = inliers.head(50).to_html(index=False, classes="table table-striped", border=0)
    in_csv  = run_dir / "inliers_no_outliers.csv"
    in_xlsx = run_dir / "inliers_no_outliers.xlsx"
    out_csv = run_dir / "only_outliers.csv"
    out_xlsx = run_dir / "only_outliers.xlsx"
    sanitize_for_spreadsheet_export(inliers).to_csv(in_csv, index=False)
    sanitize_for_spreadsheet_export(outliers).to_csv(out_csv, index=False)
    in_xlsx.write_bytes(_to_excel_bytes(inliers))
    out_xlsx.write_bytes(_to_excel_bytes(outliers))

    session["xgbod_files"] = {
        "res_csv": str(res_csv), "res_xlsx": str(res_xlsx),
        "in_csv": str(in_csv), "in_xlsx": str(in_xlsx),
        "out_csv": str(out_csv), "out_xlsx": str(out_xlsx),
        "rows": len(res_df), "n_out": int(outliers.shape[0]),
        "thr": float(thr)
    }


    return render_template("services/outlier.html",
                           artifacts=a, ready=True, step="results",
                           info_msgs=info_msgs, thr=thr,
                           rows=len(res_df), n_out=int(outliers.shape[0]),
                           outliers_preview_html=outliers_preview_html,
                           inliers_preview_html=inliers_preview_html)


@ui_bp.get("/services/outlier/download/<what>")
def outlier_download(what):
    files = session.get("xgbod_files") or {}
    mapping = {
        "results.csv": "res_csv", "results.xlsx": "res_xlsx",
        "inliers.csv": "in_csv",  "inliers.xlsx": "in_xlsx",
        "outliers.csv": "out_csv","outliers.xlsx": "out_xlsx",
    }
    key = mapping.get(what)
    path = files.get(key) if key else None
    if not path or not os.path.exists(path):
        return "Nothing to download. Please process a file first.", 404

    mime = "text/csv" if what.endswith(".csv") else \
           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return send_file(path, as_attachment=True, download_name=what, mimetype=mime)

# -------------------------------------------------------------------
# Efficient Labelling
# -------------------------------------------------------------------
@ui_bp.get("/services/labelling")
def labelling_redirect():
    target = settings.LABELLING_SERVICES_URL
    return redirect(target)
    

# -------------------------------------------------------------------
# Image Deblurring
# -------------------------------------------------------------------
@ui_bp.get("/services/deblurring")
def deblurring_redirect():
    target = settings.DEBLURRING_SERVICE_URL
    return redirect(target)
