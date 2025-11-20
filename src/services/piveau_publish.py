import os, hashlib, mimetypes, time
from pathlib import Path
from datetime import datetime, timezone,timedelta
from minio import Minio
from minio.error import S3Error
import requests
import json


def _bool(v): return str(v).strip().lower() in ("1","true","yes","on")

def _minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=_bool(os.getenv("MINIO_SECURE", "true")),
    )

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def upload_to_minio(local_path: str, object_name: str = None,
                    content_type: str = None,
                    expires_seconds: int = 7*24*3600):
    """
    Uploads local_path to MinIO and returns (object_name, presigned_url, size, sha256, content_type).
    """
    client = _minio_client()
    bucket = os.getenv("MINIO_BUCKET")
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(local_path)

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    if not object_name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        object_name = f"{ts}/{p.name}"

    if not content_type:
        content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    size = p.stat().st_size
    sha = _sha256(p)

    client.fput_object(bucket, object_name, str(p), content_type=content_type)

    # Use timedelta for recent MinIO SDKs; fall back to int for older ones
    try:
        url = client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=int(expires_seconds))
        )
    except TypeError:
        # older minio versions accept an int
        url = client.presigned_get_object(bucket, object_name, expires=int(expires_seconds))

    return {
        "bucket": bucket,
        "object": object_name,
        "url": url,
        "size": size,
        "sha256": sha,
        "content_type": content_type,
    }

def build_dcat_json(title: str, description: str, download_url: str, media_type: str, size: int, sha256: str):
    """
    Minimal DCAT-AP-ish JSON-LD for a single-file distribution in the given catalogue.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "@context": "https://www.w3.org/ns/dcat.jsonld",
        "@type": "dcat:Dataset",
        "dct:title": title,
        "dct:description": description,
        "dct:issued": now_iso,
        "dct:modified": now_iso,
        "dct:publisher": {"@type": "foaf:Agent", "foaf:name": "AI-Allianz"},
        "dcat:keyword": ["data quality", "ai-allianz", "csv"],
        "dct:license": {"@id": "https://creativecommons.org/licenses/by/4.0/"},
        # one distribution → your MinIO URL
        "dcat:distribution": [{
            "@type": "dcat:Distribution",
            "dct:title": f"{title} – Results",
            "dcat:mediaType": media_type,
            "dcat:byteSize": size,
            "spdx:checksum": {
                "@type": "spdx:Checksum",
                "spdx:algorithm": "sha256",
                "spdx:checksumValue": sha256
            },
            "dcat:downloadURL": download_url
        }]
    }

def register_dataset_in_piveau(dcat_json: dict) -> dict:
    """
    Registers the dataset in Piveau Hub-Repo for the given catalogue.
    Assumes HLRS provided a non-Keycloak auth (e.g., service token or API key).
    """
    base = os.getenv("PIVEAU_BASE").rstrip("/")
    catalog = os.getenv("PIVEAU_CATALOG_ID", "dataservices")
    scheme = os.getenv("PIVEAU_AUTH_SCHEME", "Bearer")
    token = os.getenv("PIVEAU_AUTH_TOKEN", "HuBrePokey")

    # Typical Hub-Repo path is /datasets; catalogue is chosen by header or query param in some setups.
    # HLRS uses 'dataservices' as target catalogue – they may bind it server-side.
    url = f"{base}/datasets"

    headers = {"Content-Type": "application/ld+json"}
    if token:
        if scheme.lower() == "basic":
            headers["Authorization"] = f"Basic {token}"
        elif scheme.lower() == "apikey":
            headers["X-API-Key"] = token
        else:
            headers["Authorization"] = f"{scheme} {token}"

    resp = requests.post(url, headers=headers, data=json.dumps(dcat_json), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Piveau registration failed [{resp.status_code}]: {resp.text}")

    return {
        "status": resp.status_code,
        "location": resp.headers.get("Location", ""),
        "body": resp.json() if "application/json" in resp.headers.get("Content-Type","") else resp.text
    }

def publish_result(local_result_path: str, dataset_title: str, dataset_desc: str, media_type: str = None) -> dict:
    """
    One-call publish: upload to MinIO → build DCAT → register in Piveau.
    Returns dict with useful fields to show in UI.
    """
    up = upload_to_minio(local_result_path, content_type=media_type)
    dcat = build_dcat_json(
        title=dataset_title,
        description=dataset_desc,
        download_url=up["url"],
        media_type=up["content_type"],
        size=up["size"],
        sha256=up["sha256"],
    )
    reg = register_dataset_in_piveau(dcat)

    return {
        "object": up["object"],
        "bucket": up["bucket"],
        "artifact_url": up["url"],
        "dataset_id": reg.get("location") or reg.get("body"),
        "dcat": dcat,
    }
