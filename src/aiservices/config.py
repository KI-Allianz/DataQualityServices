# src/aiservices/config.py
import os
from dataclasses import dataclass

APP_ROOT = os.getenv("APP_ROOT", "/app")  # set by Docker WORKDIR

def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(APP_ROOT, path)

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
    artifacts_dir: str = _abs(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))

    # Request limits (sane defaults)
    max_json_mb: int = int(os.getenv("MAX_JSON_MB", "10"))  # 10 MB

    # External links
    datasets_url: str = os.getenv("DATASETS_URL", "https://ki-datenraum.hlrs.de/datasets?locale=de")
    catalogues_url: str = os.getenv("CATALOGUES_URL", "https://ki-datenraum.hlrs.de/catalogues?locale=de")

settings = Settings()

# Ensure output dir exists (harmless if already present)
os.makedirs(settings.output_dir, exist_ok=True)

