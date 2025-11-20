# AI Services Production Platform (AI_Services_Production_v1)

A modular, UI-based AI inference platform for **Data Quality Analysis**, **Outlier Detection**, **Efficient Lebelling(Coming soon)** and **Image Deblurring(Coming soon)** built for deployment in **HLRS infrastructure** environment, developed by Aalen University and fundated by KI-Allianz.

---
<img width="1881" height="862" alt="image" src="https://github.com/user-attachments/assets/f9e72b0a-b67b-460c-a743-8e0115b06f77" />

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
- Fully containerized using **Docker and docker-compose**
- Easy to extend with more AI services

---

## Target Repository Structure

```
AI_Services_Production_v1/
│
├── artifacts/                     # Model files for Outlier Detection
│   ├── feature_columns.json
│   ├── scaler_XGBOD.joblib
│   ├── xgbod_detector.joblib
│   └── xgbod_threshold.joblib
│
├── output/                        # Inference output files generated at runtime
│
├── src/
│   └── aiservices/
│       ├── __init__.py
│       ├── app_factory.py         # Flask app
│       ├── config.py              # Config settings
│       ├── routes.py              # API routes
│       ├── wsgi.py                # Gunicorn entrypoint
│       │
│       ├── ui/                    # UI Layer
│       │   ├── bp.py              # Blueprint routes
│       │   ├── static/images/     # UI images
│       │   └── templates/
│       │       ├── base.html
│       │       ├── index.html
│       │       └── services/
│       │           ├── data_quality.html
│       │           ├── outlier.html
│       │           ├── labelling.html
│       │           └── deblurring.html
│       │
│       └── services/              # Backend logic for each AI module
│           ├── data_quality/
│           │   ├── feature_type_inference.py
│           │   ├── data_imputation.py
│           │   ├── anomaly_detection.py
│           │   └── personalized_detection.py
│           │
│           ├── outlier_detection/
│           |   ├── xgbod_runtime.py
│           |   ├── outlier_detection.py
│           |   └── __init__.py
|           |
|           ├── Efficient Labelling/
│           |  
|           |
|           └── Image Deblurring/  
|
├── .env                           # Env variables for runtime & Piveau 
├── requirements.txt               # Package dependencies
├── Dockerfile                     # Build instructions
├── docker-compose.yml             # Multi-service orchestration
├── gunicorn.conf.py               # Server configuration
└── README.md                      # Documentation
```

---

## Local Development Setup

```bash
git clone <repo-url>
cd AI_Services_Production_v1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=src/aiservices/app_factory.py
flask run
```

View the UI at:

```
http://localhost:5000
```

---

## Docker Deployment (Recommended)

### Build image
```bash
docker build -t ai-services .
```

### Run
```bash
docker run -p 8000:8000 ai-services
```

View the UI at:

```
http://localhost:8000
```

---

## Docker Compose Deployment

```bash

docker-compose up --build
```

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
OUTPUT_DIR=./output
XGBOD_ARTIFACTS_DIR=./artifacts
PIVEAU_TOKEN=<optional>
HUB_STORE_URL=<optional>
HUB_STORE_BUCKET=ai-results
```

---

## API

For DataQuality API:http://localhost:8000/services/data-quality
For Outlier Detection API:http://localhost:8000/services/outlier

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
| **Bhuvneshwar Bajpeyee** | Aalen University |
| **Miroslav** | Hosting & infrastructure support |

---

## How to Contribute

1. Fork this repo
2. Create your branch: `git checkout -b feature/new-service`
3. Commit: `git commit -m 'Add new service'`
4. Push: `git push origin feature/new-service`
5. Create a Pull Request

---
