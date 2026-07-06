# AI Services Production Platform (AI_Services_Production_BaseVersion)

A unified AI platform offering **Data Quality Analysis**, **Outlier Detection**, **Efficient Lebelling** and **Image Deblurring** services and deployment been done in **HLRS infrastructure** environment, developed by Aalen University and funded by KI-Allianz.

![alt text](image-1.png)

---

## Features

- **Four AI services integrated**:
  - **Data Quality AI:** Feature type detection, imputation, anomaly detection, personalized summary
  - **Outlier Detection (XGBOD):** ML-based outlier detection for tabular data
  - **Efficient Lebelling:** Large image datasets are often unlabeled or partially labeled, making them unusable for supervised ML workflows.
  - **Image Deblurring:** Restores blurred images while preserving resolution, format, and EXIF metadata.
- **Wizard-based UI with "Back / Next / Reset" navigation**
- **REST API routes available** 
- Supports **CSV/XLSX uploads**, runs inference, and exports results
- Integration with **Piveau Hub** for dataset publishing.
- Security hardening for uploads, JSON APIs, spreadsheet exports, secrets, Flask runtime configuration, and model artifact loading.
- Fully containerized using **Docker and docker-compose**
- Easy to extend with more AI services

---

---

## Local Docker Development Setup(Powershell) for Window & Linux
# 1. Clone repo
git clone <repo-url>
cd AI_Services_Production_BaseVersion
# 2. Check Python version (should be 3.11+)
python --version      # For Window
python3 --version     # For Linux
# 3. Create & activate virtualenv
# For Window
py -3.11 -m venv venv   
.\venv\bin\activate     
# For Linux
python3 -m venv venv     
source venv/bin/activate 
# 4. Install dependencies (Window & Linux)
pip install --upgrade pip
pip install -r requirements.txt

### Build & run docker stack (Window & Linux)
docker compose down         # stop/remove old containers if any
docker compose build        # build ai-services + efficient-labelling
docker compose up -d        # start in background
docker compose logs -f      # follow logs from both services

View the Main UI at:

```
http://localhost:8000
```

### Linux/NVIDIA GPU for DataQuality

The default DataQuality compose file remains macOS-friendly. On Linux hosts with NVIDIA GPUs, install the NVIDIA driver and NVIDIA Container Toolkit, verify `nvidia-smi` on the host, then start the DataQuality backend with the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

For the nested DataQuality-only stack:

```bash
cd dataquality-aiservices
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Check `http://localhost:8503/status`; the response includes `gpu.cuda_available` and detected GPU names when the backend container can access NVIDIA.

---

## AI Services Overview

### Data Quality AI Service
Descriptions:
- Automated Feature Type Inference
The automated feature type inference service analyzes each column in a dataset and assigns it a semantic type (such as numeric, categorical, sentence, URL, list, embedded-number, context-specific, not-generalizable or datetime). This enables downstream preprocessing and machine learning components to handle each feature appropriately. It currently distinguishes between nine different feature classes and allows users to review and manually adjust the inferred types where necessary. Link: …

- Detection of Personal Data
The personal data detection service scans structured datasets to identify columns and fields that likely contain personal or sensitive information (such as names, contact details, or identifiers). It analyses column names, descriptions, and values to flag potentially privacy-relevant attributes so they can be handled appropriately in privacy-preserving and compliant data preparation workflows. Link: https://arxiv.org/abs/2506.22305 

-Automated Imputation of Tabular Data
The imputation service automatically handles missing values in tabular datasets, for both single- and multi-column missing data, using mean and mode imputation. Extensive evaluation showed that simple mean/mode imputation offers competitive accuracy (within roughly ±3% of advanced methods such as autoencoders or random forests) at a fraction of the computational cost. Consequently, mean/mode imputation is used as a robust default to provide complete, consistently imputed data for downstream analyses and machine learning models. Link: https://dl.acm.org/doi/full/10.1145/3643643 

- Anomaly Detection
The anomaly detection service automatically finds unusual or inconsistent data points in tabular and time-series datasets. It highlights values and patterns that deviate from expected behaviour, helping to reveal potential errors, sensor faults, or rare events as part of the overall data-quality process.  Link: https://github.com/yzhao062/pyod  / https://arxiv.org/abs/2201.07284 

Steps:
1. Upload CSV/XLSX
2. Select target column (optional)
3. Run pipeline (Feature Type → Imputation → Anomaly Detection →Personalized_detection→ Summary)
4. Download outputs or publish to Piveau

Key files:
- `feature_type_inference.py`
- `data_imputation.py`
- `anomaly_detection.py`
- `personalized_detection.py`

---

### Outlier Detection (XGBOD)
- Upload a CSV/XLSX
- Uses pretrained XGBOD model (`artifacts/`)
- Outputs:
  - `results.csv`
  - `inliers_no_outliers.csv`
  - `only_outliers.csv`

Key module: `xgbod_runtime.py`

---

## Environment Variables

Create `.env` file:
```bash
SECRET_KEY=<generate-with-python-secrets-token-hex>
OUTPUT_DIR=./output
XGBOD_ARTIFACTS_DIR=./artifacts

# JSON API safety limits
MAX_RECORDS=50000
MAX_COLUMNS=500
MAX_CELL_CHARS=10000

# DataQuality upload safety limits
DATAQUALITY_MAX_UPLOAD_ROWS=50000
DATAQUALITY_MAX_UPLOAD_COLUMNS=500

# Flask runtime
FLASK_DEBUG=false

# Piveau / MinIO publishing
MINIO_ENDPOINT=<optional>
MINIO_ACCESS_KEY=<optional>
MINIO_SECRET_KEY=<optional>
MINIO_BUCKET=<optional>
PIVEAU_BASE=<optional>
PIVEAU_AUTH_TOKEN=<optional>
PIVEAU_AUTH_SCHEME=Bearer
PIVEAU_CATALOG_ID=dataservices
```

Generate a local development secret with:

```bash
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))" > .env
```

Do not commit `.env` files or generated Flask session files.

---

## Security Notes

The services include several OWASP-oriented hardening measures:

- Secrets are read from environment variables. The Piveau publisher no longer contains a hardcoded fallback token and fails clearly when `PIVEAU_BASE` or `PIVEAU_AUTH_TOKEN` is missing.
- Flask debug mode is disabled by default and must be explicitly enabled with `FLASK_DEBUG=true` for local debugging.
- JSON API requests are validated before pandas `DataFrame` creation using `MAX_RECORDS`, `MAX_COLUMNS`, and `MAX_CELL_CHARS`.
- DataQuality uploads are capped with `DATAQUALITY_MAX_UPLOAD_ROWS` and `DATAQUALITY_MAX_UPLOAD_COLUMNS`.
- CSV, XLSX, and ZIP exports neutralize spreadsheet formulas so values starting with characters such as `=`, `+`, `-`, or `@` are exported as text.
- Image Deblurring ZIP uploads are validated for safe paths, supported image types, image count, compressed/uncompressed size, suspicious compression ratios, and maximum pixel count before processing.
- Flask session folders are ignored by Git, and committed session files were removed.
- User-provided "contains" filters use literal string matching instead of regular expressions.
- XGBOD model artifacts are loaded only from known files under the configured artifacts directory.

---

## API
```
For DataQuality API:http://localhost:8503
```
```
For Outlier Detection API:http://localhost:8000/services/outlier
```
```
For Image Deblurring API:http://localhost:8502
```
```
For Efficient Labelling API:http://localhost:8501
```

---

## Publishing to Piveau Hub 

Handled by:
```
src/services/piveau_publish.py
```

Only enabled if:
- Token provided in `.env`
- Proper MinIO / Piveau variables configured

---


## License

```
Apache License 2.0
```

---

## Maintainers

| Name | Affiliation |
|------|-------------|
| **Bhuvneshwar Bajpeyee** | AI Services Development & Integration Ownership | Aalen University |
| **Petros Tsialis & Albert Agisha** | DataQuality Services Owner | Aalen University |
| **Dima Al-Obaidi & Felix Gerschner** | Efficient Labelling Service Owner | Aalen University |
| **Niloofar Kalashtari** | Outlier Detection Service Owner | Aalen University |
| **Patrick Krawczyk** | Image deblurringg Service Owner | Aalen University |

---

## How to Contribute

1. Fork this repo
2. Create your branch: `git checkout -b feature/new-service`
3. Commit: `git commit -m 'Add new service'`
4. Push: `git push origin feature/new-service`
5. Create a Pull Request

---
