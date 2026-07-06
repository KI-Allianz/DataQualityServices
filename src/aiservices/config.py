# src/aiservices/config.py
import os
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(os.getenv("APP_ROOT", "/app")).resolve()

def _abs(path: str) -> str:
    p = Path(path)
    return p if p.is_absolute() else APP_ROOT / p

@dataclass
class Settings:
    """Application configuration settings for production build."""

    # General metadata
    api_title: str = os.getenv("API_TITLE", "AI Services API")
    api_version: str = os.getenv("API_VERSION", "0.1.0")

    # Server settings
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Directories (absolute inside container)
    output_dir: str = _abs(os.getenv("OUTPUT_DIR", "/app/output"))
    upload_dir: str = _abs(os.getenv("UPLOAD_DIR", "/app/uploads"))
    artifacts_dir: str = _abs(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))
    
    output_retention_days: int = int(os.getenv("OUTPUT_RETENTION_DAYS", "7"))
    upload_retention_days: int = int(os.getenv("UPLOAD_RETENTION_DAYS", "7"))

    # Request limits (sane defaults)
    max_json_mb: int = int(os.getenv("MAX_JSON_MB", "10"))  

    # External links
    datasets_url: str = os.getenv("DATASETS_URL", "https://ki-datenraum.hlrs.de/datasets?locale=de")
    catalogues_url: str = os.getenv("CATALOGUES_URL", "https://ki-datenraum.hlrs.de/catalogues?locale=de")

    LABELLING_SERVICES_URL: str = os.getenv("LABELLING_SERVICES_URL", "http://localhost:8501")
    DEBLURRING_SERVICE_URL: str = os.getenv("DEBLURRING_SERVICE_URL", "http://localhost:8502")
    DATAQUALITY_SERVICE_URL: str = os.getenv("DATAQUALITY_SERVICE_URL", "http://localhost:8503")


settings = Settings()

os.makedirs(settings.output_dir, exist_ok=True)
