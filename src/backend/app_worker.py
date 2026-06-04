"""
Cloudflare Workers-compatible FastAPI app for Atlético Intelligence.

Key differences from main.py (local server):
  - No ML dependencies (torch/ultralytics/opencv not available in Workers)
  - No local filesystem (no Path writes)
  - Job state stored in Cloudflare KV (injected via request.scope["env"])
  - Video uploads staged as base64 in KV, pushed to Kaggle dataset via REST API
  - Kaggle SDK replaced with httpx REST calls (kaggle pip package not available in Workers)
  - LOCAL_PROCESSING mode removed (Workers can't run YOLO)
"""

import json
import uuid
import base64
from datetime import datetime
from typing import Optional

import js
from pyodide.ffi import to_js as _to_js
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Atlético Intelligence API",
    description="Offside and goal detection via computer vision (Cloudflare Workers)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# KV helpers  (env is injected by Workers runtime via request.scope["env"])
# ---------------------------------------------------------------------------

async def kv_get(env, key: str) -> Optional[dict]:
    raw = await env.JOB_STATE.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def kv_put(env, key: str, value: dict, ttl: int = 86400 * 30) -> None:
    await env.JOB_STATE.put(key, json.dumps(value), expirationTtl=ttl)


async def kv_delete(env, key: str) -> None:
    await env.JOB_STATE.delete(key)


# ---------------------------------------------------------------------------
# HTTP helpers using js.fetch (Cloudflare Workers native — zero bundle cost)
# ---------------------------------------------------------------------------


def _js_init(opts: dict):
    """Convert Python dict to JS RequestInit object."""
    return _to_js(opts, dict_converter=js.Object.fromEntries)


async def _fetch(url: str, method: str = "GET", headers: dict = None, body=None) -> tuple[int, str]:
    """Async HTTP call via Workers-native js.fetch."""
    opts: dict = {"method": method}
    if headers:
        opts["headers"] = headers
    if body is not None:
        opts["body"] = body
    response = await js.fetch(url, _js_init(opts))
    text = await response.text()
    return response.status, text


async def _fetch_multipart(
    url: str,
    fields: dict,
    file_data: bytes,
    file_field: str,
    filename: str,
    headers: dict = None,
) -> tuple[int, str]:
    """POST multipart/form-data using js.FormData + js.Blob."""
    form = js.FormData.new()
    for k, v in fields.items():
        form.append(k, str(v))
    blob = js.Blob.new(
        _to_js([bytearray(file_data)]),
        _js_init({"type": "application/octet-stream"}),
    )
    form.append(file_field, blob, filename)
    opts: dict = {"method": "POST", "body": form}
    if headers:
        opts["headers"] = headers
    response = await js.fetch(url, _js_init(opts))
    text = await response.text()
    return response.status, text


# ---------------------------------------------------------------------------
# Kaggle REST helpers (replaces kaggle Python SDK)
# ---------------------------------------------------------------------------

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


def _kaggle_headers(kaggle_key: str) -> dict:
    """Build Authorization header. Supports KGAT Bearer and legacy Basic."""
    if kaggle_key.startswith("KGAT_"):
        return {"Authorization": f"Bearer {kaggle_key}"}
    return {}


async def kaggle_push_dataset_version(
    username: str,
    kaggle_key: str,
    dataset_name: str,
    job_id: str,
    video_bytes: bytes,
    video_filename: str,
    incident_type: str,
    frame_number: Optional[int],
) -> tuple[bool, str]:
    """Push a new dataset version to Kaggle containing the job video + config."""
    headers = _kaggle_headers(kaggle_key)

    current_job = {
        "job_id": job_id,
        "video_filename": video_filename,
        "incident_type": incident_type,
        "frame_number": frame_number,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    url = f"{KAGGLE_API_BASE}/datasets/{username}/{dataset_name}/versions"

    # Push video + current_job.json together as multipart
    status, text = await _fetch_multipart(
        url=url,
        fields={"versionNotes": f"Job {job_id[:8]} - {incident_type}", "deletePreviousVersions": "false", "convertToCsv": "false"},
        file_data=video_bytes,
        file_field="file",
        filename=video_filename,
        headers=headers,
    )

    if status in (200, 201):
        # Also push current_job.json
        await _fetch_multipart(
            url=url,
            fields={"versionNotes": f"Job config {job_id[:8]}", "deletePreviousVersions": "false", "convertToCsv": "false"},
            file_data=json.dumps(current_job).encode(),
            file_field="file",
            filename="current_job.json",
            headers=headers,
        )
        return True, f"Dataset version created for job {job_id[:8]}"

    if status == 404:
        return await kaggle_create_dataset(
            username, kaggle_key, dataset_name, job_id,
            video_bytes, video_filename, incident_type, frame_number,
        )

    return False, f"Dataset push failed [{status}]: {text[:300]}"


async def kaggle_create_dataset(
    username: str,
    kaggle_key: str,
    dataset_name: str,
    job_id: str,
    video_bytes: bytes,
    video_filename: str,
    incident_type: str,
    frame_number: Optional[int],
) -> tuple[bool, str]:
    """Create dataset for first-time setup (JSON API with base64 files)."""
    headers = {**_kaggle_headers(kaggle_key), "Content-Type": "application/json"}

    current_job = {
        "job_id": job_id,
        "video_filename": video_filename,
        "incident_type": incident_type,
        "frame_number": frame_number,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    payload = {
        "title": "Athletic Intelligence Dataset",
        "slug": dataset_name,
        "ownerSlug": username,
        "licenses": [{"nameNullable": "CC0-1.0"}],
        "isPrivate": True,
        "files": [
            {
                "fileName": "current_job.json",
                "content": base64.b64encode(json.dumps(current_job).encode()).decode(),
            }
        ],
    }

    url = f"{KAGGLE_API_BASE}/datasets/create/new"
    status, text = await _fetch(url, method="POST", headers=headers, body=json.dumps(payload))

    if status in (200, 201):
        return True, "Dataset created (initial)"
    return False, f"Dataset create failed [{status}]: {text[:300]}"


async def kaggle_push_kernel(
    username: str,
    kaggle_key: str,
    kernel_slug: str,
    dataset_name: str,
    job_id: str,
) -> tuple[bool, str]:
    """Push kernel to trigger a new run on Kaggle."""
    headers = {**_kaggle_headers(kaggle_key), "Content-Type": "application/json"}

    payload = {
        "id": f"{username}/{kernel_slug}",
        "title": "Athletic Intelligence - Offside & Goal Detection",
        "code_file": "kaggle_main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_data_sources": [f"{username}/{dataset_name}"],
    }

    url = f"{KAGGLE_API_BASE}/kernels/push"
    status, text = await _fetch(url, method="POST", headers=headers, body=json.dumps(payload))

    if status in (200, 201):
        ref = json.loads(text).get("ref", f"{username}/{kernel_slug}")
        return True, str(ref)
    return False, f"Kernel push failed [{status}]: {text[:300]}"


async def kaggle_kernel_status(
    username: str,
    kaggle_key: str,
    kernel_slug: str,
) -> tuple[str, str]:
    """Return (status, failure_message). Status: queued|running|complete|failed|unknown."""
    headers = _kaggle_headers(kaggle_key)
    url = f"{KAGGLE_API_BASE}/kernels/{username}/{kernel_slug}/status"
    status, text = await _fetch(url, method="GET", headers=headers)

    if status != 200:
        return "unknown", f"Status check failed [{status}]"

    data = json.loads(text)
    raw = (data.get("status") or "").lower()
    status_map = {
        "queued": "queued",
        "running": "running",
        "complete": "complete",
        "error": "failed",
        "cancelrequested": "running",
        "cancelacknowledged": "failed",
        "newscript": "queued",
    }
    return status_map.get(raw, "unknown"), data.get("failureMessage") or ""


async def kaggle_download_verdict(
    username: str,
    kaggle_key: str,
    kernel_slug: str,
    job_id: str,
) -> tuple[Optional[dict], str]:
    """Download verdict.json from kernel output."""
    headers = _kaggle_headers(kaggle_key)
    url = f"{KAGGLE_API_BASE}/kernels/{username}/{kernel_slug}/output"
    status, text = await _fetch(url, method="GET", headers=headers)

    if status != 200:
        return None, f"Output download failed [{status}]: {text[:200]}"

    data = json.loads(text)
    files = data.get("files") or []
    for f in files:
        if f.get("fileName", "").endswith("verdict.json"):
            content_url = f.get("url") or f.get("downloadUrl")
            if content_url:
                vs, vt = await _fetch(content_url, headers=headers)
                if vs == 200:
                    return json.loads(vt), "OK"
            if f.get("content"):
                return json.loads(f["content"]), "OK"

    return None, f"verdict.json not found in output. Files: {[f.get('fileName') for f in files]}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Atlético Intelligence API",
        "version": "1.0.0",
        "status": "online",
        "runtime": "Cloudflare Workers",
    }


@app.get("/health", tags=["health"])
async def health(request: Request):
    env = request.scope.get("env")
    kaggle_configured = bool(getattr(env, "KAGGLE_KEY", None)) if env else False
    return {
        "status": "healthy",
        "kaggle_configured": kaggle_configured,
        "runtime": "cloudflare-workers",
    }


@app.post("/upload", response_model=UploadResponse, tags=["upload"])
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    incident_type: str = Form("offside"),
    frame_number: Optional[int] = Form(None),
):
    """Upload video and queue it for Kaggle kernel processing."""
    env = request.scope.get("env")
    if not env:
        raise HTTPException(status_code=500, detail="Runtime env not available")

    kaggle_username = getattr(env, "KAGGLE_USERNAME", None)
    kaggle_key = getattr(env, "KAGGLE_KEY", None)
    dataset_name = getattr(env, "KAGGLE_DATASET_NAME", "athletic-intelligence-dataset")
    kernel_slug = getattr(env, "KAGGLE_KERNEL_SLUG", "athletic-intelligence-offside-goal-detection")

    if not kaggle_username or not kaggle_key:
        raise HTTPException(status_code=503, detail="Kaggle credentials not configured")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    valid_types = {"offside", "goal"}
    if incident_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"incident_type must be one of {valid_types}")

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > 500:
        raise HTTPException(status_code=413, detail=f"File too large: {file_size_mb:.1f}MB > 500MB")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "mp4"
    if ext not in {"mp4", "avi", "mov", "mkv"}:
        raise HTTPException(status_code=400, detail=f"Invalid format: .{ext}")

    job_id = str(uuid.uuid4())
    video_filename = f"{job_id}.{ext}"

    job = {
        "job_id": job_id,
        "video_filename": file.filename,
        "status": "uploading",
        "progress": 5,
        "created_at": datetime.utcnow().isoformat(),
        "incident_type": incident_type,
        "frame_number": frame_number,
        "kaggle_kernel_slug": kernel_slug,
        "verdict": None,
        "confidence": None,
        "error": None,
        "completed_at": None,
    }
    await kv_put(env, f"job:{job_id}", job)

    success, msg = await kaggle_push_dataset_version(
        username=kaggle_username,
        kaggle_key=kaggle_key,
        dataset_name=dataset_name,
        job_id=job_id,
        video_bytes=file_bytes,
        video_filename=video_filename,
        incident_type=incident_type,
        frame_number=frame_number,
    )

    if not success:
        job["status"] = "failed"
        job["error"] = f"Dataset push failed: {msg}"
        await kv_put(env, f"job:{job_id}", job)
        raise HTTPException(status_code=502, detail=job["error"])

    job["status"] = "processing"
    job["progress"] = 30
    await kv_put(env, f"job:{job_id}", job)

    ok, kernel_ref = await kaggle_push_kernel(
        username=kaggle_username,
        kaggle_key=kaggle_key,
        kernel_slug=kernel_slug,
        dataset_name=dataset_name,
        job_id=job_id,
    )

    if not ok:
        job["status"] = "failed"
        job["error"] = f"Kernel push failed: {kernel_ref}"
        await kv_put(env, f"job:{job_id}", job)
        raise HTTPException(status_code=502, detail=job["error"])

    job["status"] = "queued"
    job["progress"] = 40
    job["kaggle_kernel_ref"] = kernel_ref
    await kv_put(env, f"job:{job_id}", job)

    return UploadResponse(
        job_id=job_id,
        status="queued",
        message=f"Video uploaded and kernel queued. Poll /status/{job_id} for progress.",
    )


@app.get("/status/{job_id}", response_model=StatusResponse, tags=["status"])
async def get_status(job_id: str, request: Request):
    """Get job status. If kernel is running, polls Kaggle for latest state."""
    env = request.scope.get("env")
    if not env:
        raise HTTPException(status_code=500, detail="Runtime env not available")

    job = await kv_get(env, f"job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] in ("queued", "processing"):
        kaggle_username = getattr(env, "KAGGLE_USERNAME", None)
        kaggle_key = getattr(env, "KAGGLE_KEY", None)
        kernel_slug = job.get("kaggle_kernel_slug", getattr(env, "KAGGLE_KERNEL_SLUG", ""))

        if kaggle_username and kaggle_key and kernel_slug:
            k_status, fail_msg = await kaggle_kernel_status(kaggle_username, kaggle_key, kernel_slug)

            if k_status == "complete":
                verdict_data, dl_msg = await kaggle_download_verdict(
                    kaggle_username, kaggle_key, kernel_slug, job_id
                )
                if verdict_data:
                    job["status"] = "completed"
                    job["progress"] = 100
                    job["verdict"] = verdict_data.get("verdict")
                    job["confidence"] = float(verdict_data.get("confidence", 0.0))
                    job["completed_at"] = datetime.utcnow().isoformat()
                else:
                    job["status"] = "failed"
                    job["error"] = f"Output download failed: {dl_msg}"

            elif k_status == "failed":
                job["status"] = "failed"
                job["error"] = fail_msg or "Kernel execution failed on Kaggle"

            elif k_status == "running":
                job["status"] = "processing"
                job["progress"] = min(job.get("progress", 40) + 5, 90)

            await kv_put(env, f"job:{job_id}", job)

    return StatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        verdict=job.get("verdict"),
        confidence=job.get("confidence"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
    )


@app.get("/jobs", tags=["jobs"])
async def list_jobs(request: Request):
    """List jobs index (KV index key)."""
    env = request.scope.get("env")
    if not env:
        raise HTTPException(status_code=500, detail="Runtime env not available")

    index_raw = await env.JOB_STATE.get("jobs:index")
    if not index_raw:
        return {"count": 0, "jobs": [], "note": "No job index found. Submit a job first."}

    job_ids = json.loads(index_raw)
    jobs = []
    for jid in job_ids[-20:]:
        j = await kv_get(env, f"job:{jid}")
        if j:
            jobs.append({
                "job_id": j["job_id"],
                "video_filename": j.get("video_filename"),
                "status": j["status"],
                "progress": j.get("progress", 0),
                "verdict": j.get("verdict"),
                "created_at": j.get("created_at"),
            })

    return {"count": len(jobs), "jobs": list(reversed(jobs))}


@app.delete("/jobs/{job_id}", tags=["jobs"])
async def delete_job(job_id: str, request: Request):
    env = request.scope.get("env")
    if not env:
        raise HTTPException(status_code=500, detail="Runtime env not available")

    job = await kv_get(env, f"job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    await kv_delete(env, f"job:{job_id}")
    return {"message": f"Job {job_id} deleted"}
