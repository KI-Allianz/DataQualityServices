# bp.py — HTML UI blueprint 
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


# ---- use package-relative imports (works when 'aiservices' is a package) ----
from src.services.data_quality.feature_type_inference import detect_feature_types
from src.services.data_quality.data_imputation import impute_missing
from src.services.data_quality.anomaly_detection import anomaly_score
from src.services.data_quality.personalized_detection import apply_rules
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
    return render_template("index.html")


# -------------------------------------------------------------------
# Helpers / State
# -------------------------------------------------------------------
OUTPUT_DIR = Path(settings.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _dq_state():
    st = session.get("data_quality_state")
    if not st:
        st = {}
        session["data_quality_state"] = st
    return st

def _as_bool(v):
    """Parse checkbox/select truthy values reliably."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    return v in ("1", "true", "on", "yes", "y")

# ----- Wizard helpers -----
def _dq_get_step() -> int:
    st = _dq_state()
    step = int(st.get("dq_step", 0))
    # auto-advance to step 1 if a file exists but step is 0
    if step == 0 and st.get("csv_path"):
        step = 1
        st["dq_step"] = 1
        session.modified = True
    return max(0, min(step, 2))  # clamp 0..2

def _dq_set_step(n: int):
    st = _dq_state()
    st["dq_step"] = max(0, min(int(n), 2))
    session.modified = True
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
def data_quality_page():
    """Step 0: upload page, or show previous preview if present."""
    st = _dq_state()
    step = _dq_get_step()
    df_preview, cols = None, []
    if st.get("csv_path"):
        #step = 1
        try:
            df = pd.read_csv(st["csv_path"])
            df_preview = df.head(12).to_html(index=False, classes="table", border=0)
            cols = list(df.columns)
        except Exception as e:
            st.clear()
            _dq_set_step(0)
            return render_template("services/data_quality.html", step=0,
                                   error=f"Failed to read previous upload: {e}")
    return render_template("services/data_quality.html", step=step,
                           df_preview=df_preview, columns=cols)

@ui_bp.post("/services/data-quality/next")
def data_quality_next():
    st = _dq_state()
    step = _dq_get_step()

    # Guards: only advance if prerequisites are satisfied
    if step == 0:
        if not st.get("csv_path"):
            # no file yet → stay on upload
            return redirect(url_for("ui.data_quality_page"))
        _dq_set_step(1)
    elif step == 1:
        # You may require that processing ran at least once:
        # if not st.get("fti_csv") and not st.get("imp_csv") and not st.get("anom_csv"):
        #     return redirect(url_for("ui.data_quality_page"))
        _dq_set_step(2)
    # step 2 → stay at results
    return redirect(url_for("ui.data_quality_page"))

@ui_bp.post("/services/data-quality/back")
def data_quality_back():
    step = _dq_get_step()
    if step > 0:
        _dq_set_step(step - 1)
    return redirect(url_for("ui.data_quality_page"))

@ui_bp.post("/services/data-quality/reset")
def data_quality_reset():
    # clear only the Data Quality state
    st = session.get("data_quality_state")
    if st:
        # (optional) you could remove temp files pointed by st[...] here
        session.pop("data_quality_state", None)
    session.pop("dq_step", None)
    session.modified = True
    return redirect(url_for("ui.data_quality_page"))

@ui_bp.post("/services/data-quality/upload")
def data_quality_upload():
    st = _dq_state()
    up = request.files.get("data_file")
    if not up or not up.filename:
        return render_template("services/data_quality.html", step=0,
                               error="Please upload a CSV or Excel file.")
    try:
        run_id = uuid.uuid4().hex
        csv_path = OUTPUT_DIR / f"dq_{run_id}.csv"

        # Accept CSV/XLSX; normalize to CSV on disk
        if up.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(up)
        else:
            df = pd.read_csv(up)
        df.to_csv(csv_path, index=False)

        st.clear()
        st.update({"run_id": run_id, "csv_path": str(csv_path), "target_col": None})
        session.modified = True

        df_preview = df.head(12).to_html(index=False, classes="table", border=0)
        return render_template("services/data_quality.html", step=1,
                               df_preview=df_preview, columns=list(df.columns))
    except Exception as e:
        return render_template("services/data_quality.html", step=0,
                               error=f"Could not read file: {e}")
@ui_bp.post("/services/data-quality/run")
def data_quality_run():
    st = _dq_state()
    run_id = st.get("run_id")
    csv_path = st.get("csv_path")
    if not run_id or not csv_path or not os.path.exists(csv_path):
        return redirect(url_for("ui.data_quality_page"))

    target = (request.form.get("target_col") or "").strip()
    st["target_col"] = target

    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        return render_template("services/data_quality.html", step=0,
                               error=f"Reload failed: {e}")

    # 1) Feature types
    fti_df, fti_err = None, None
    try:
        types = detect_feature_types(df_raw)
        fti_df = pd.DataFrame({"column": list(types.keys()), "type": list(types.values())})
        fti_df.to_csv(OUTPUT_DIR / f"fti_{run_id}.csv", index=False)
        st["has_fti"] = True
    except Exception as e:
        fti_err = str(e); st.pop("has_fti", None)

    # 2) Imputation
    imp_df, imp_err = None, None
    try:
        strategy = (request.form.get("strategy") or "mean").strip()
        imp_df = impute_missing(df_raw.copy(), strategy=strategy)
        imp_df.to_csv(OUTPUT_DIR / f"imputed_{run_id}.csv", index=False)
        st["has_imp"] = True
    except Exception as e:
        imp_err = str(e); st.pop("has_imp", None)

    # 3) Anomaly (IF fallback)
    anom_df, anom_err, anom_threshold = None, None, "IsolationForest (contamination=0.10)"
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        num_df = df_raw.select_dtypes(include=[np.number]).copy()
        if num_df.shape[1] == 0:
            anom_df = df_raw.copy(); anom_df["Anomaly"] = 0
        else:
            X = StandardScaler().fit_transform(num_df.values)
            scores = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)\
                        .fit_predict(X)
            anom_df = df_raw.copy(); anom_df["Anomaly"] = (scores == -1).astype(int)
        anom_df.to_csv(OUTPUT_DIR / f"anomaly_{run_id}.csv", index=False)
        st["has_anom"] = True
    except Exception as e:
        anom_err = str(e); st.pop("has_anom", None)

    # 4) Personalized preview
    pers_df, pers_msg = None, None
    try:
        if target and target in df_raw.columns:
            vc = df_raw[target].value_counts(dropna=False).reset_index()
            vc.columns = [target, "count"]
            pers_df = vc
        else:
            pers_msg = "Pick a target column to see distribution."
    except Exception as e:
        pers_msg = f"Personalized result unavailable: {e}"

    # Bundle for publish (derive from run_id)
    publish_info, publish_err = None, None
    try:
        from zipfile import ZipFile, ZIP_DEFLATED
        bundle = OUTPUT_DIR / f"dq_bundle_{run_id}.zip"
        with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as zf:
            for base in ("fti", "imputed", "anomaly"):
                p = OUTPUT_DIR / f"{base}_{run_id}.csv"
                if p.exists(): zf.write(p, arcname=p.name)
        if bundle.exists() and bundle.stat().st_size > 0:
            publish_info = publish_result(
                local_result_path=str(bundle),
                dataset_title="AI-Allianz – Data Quality Results",
                dataset_desc="Outputs: feature type inference, imputation, anomalies.",
                media_type="application/zip")
        else:
            publish_err = "No result files to publish (bundle is empty)."
    except Exception as e:
        publish_err = f"Publish failed: {e}"

    # Previews
    fti_prev = fti_df.head(12).to_html(index=False, classes='table', border=0) if fti_df is not None else None
    imp_prev = imp_df.head(12).to_html(index=False, classes='table', border=0) if imp_df is not None else None
    anom_prev = anom_df.head(12).to_html(index=False, classes='table', border=0) if anom_df is not None else None
    pers_prev = pers_df.head(20).to_html(index=False, classes='table', border=0) if pers_df is not None else None

    session.modified = True
    return render_template("services/data_quality.html",
        step=2, target=target,
        fti_preview=fti_prev, fti_err=fti_err,
        imp_preview=imp_prev, imp_err=imp_err,
        anom_preview=anom_prev, anom_err=anom_err, anom_threshold=anom_threshold,
        pers_preview=pers_prev, pers_msg=pers_msg,
        publish_info=publish_info, publish_err=publish_err)


@ui_bp.get("/services/data-quality/download/<kind>.<fmt>")
def data_quality_download(kind, fmt):
    st = session.get("data_quality_state") or {}
    run_id = st.get("run_id")
    if not run_id:
        return "No result yet.", 404
    # accept both aliases
    name_map = {
        "fti": "fti",
        "imp": "imputed",
        "imputed": "imputed",
        "anom": "anomaly",
        "anomaly": "anomaly",
    }
    base = name_map.get(kind.lower())
    #base = {"fti": "fti", "imp": "imputed", "anom": "anomaly"}.get(kind)
    if not base:
        return "Unsupported kind", 400

    csv_path = OUTPUT_DIR / f"{base}_{run_id}.csv"
    if not csv_path.exists():
        return "No result yet.", 404

    if fmt == "csv":
        return send_file(csv_path, as_attachment=True,
                         download_name=f"{base}.csv", mimetype="text/csv")
    elif fmt == "xlsx":
        df = pd.read_csv(csv_path)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name=base)
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f"{base}.xlsx",
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return "Unsupported format", 400


# -------------------------------------------------------------------
# Outlier Detection UI (XGBOD)
# -------------------------------------------------------------------
def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="results")
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
    res_df.to_csv(res_csv, index=False)
    res_xlsx.write_bytes(_to_excel_bytes(res_df))

    inliers  = res_df.loc[res_df["detected outliers"] == 0]
    outliers = res_df.loc[res_df["detected outliers"] == 1]
    in_csv  = run_dir / "inliers_no_outliers.csv"
    in_xlsx = run_dir / "inliers_no_outliers.xlsx"
    out_csv = run_dir / "only_outliers.csv"
    out_xlsx = run_dir / "only_outliers.xlsx"
    inliers.to_csv(in_csv, index=False)
    outliers.to_csv(out_csv, index=False)
    in_xlsx.write_bytes(_to_excel_bytes(inliers))
    out_xlsx.write_bytes(_to_excel_bytes(outliers))

    session["xgbod_files"] = {
        "res_csv": str(res_csv), "res_xlsx": str(res_xlsx),
        "in_csv": str(in_csv), "in_xlsx": str(in_xlsx),
        "out_csv": str(out_csv), "out_xlsx": str(out_xlsx),
        "rows": len(res_df), "n_out": int(outliers.shape[0]),
        "thr": float(thr)
    }

    preview_html = res_df.head(50).to_html(index=False, classes="table table-striped", border=0)
    return render_template("services/outlier.html",
                           artifacts=a, ready=True, step="results",
                           info_msgs=info_msgs, thr=thr,
                           rows=len(res_df), n_out=int(outliers.shape[0]),
                           preview_html=preview_html)

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
