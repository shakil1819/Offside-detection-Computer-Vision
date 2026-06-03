# Kaggle Integration — Complete Setup Guide

**Status:** Implemented  
**Date:** June 4, 2026

---

## Architecture

```
Backend (local)                 Kaggle Cloud
    │                               │
    ├─ push_job_to_dataset() ──────► dataset_create_version()
    │   copies video + config.json   adds input/{job_id}.mp4
    │   to kaggle-dataset/input/     adds input/current_job.json
    │
    ├─ push_kernel() ─────────────► kernels_push(acc='NvidiaTeslaT4')
    │   reads kaggle-kernel/         triggers new kernel run
    │   kernel-metadata.json
    │
    ├─ get_kernel_status() [loop] ► kernels_status()
    │   polls every 5s               returns QUEUED/RUNNING/COMPLETE/ERROR
    │
    └─ download_results() ────────► kernels_output()
        fetches /kaggle/working/     downloads all output files
        to outputs/{job_id}/
```

---

## One-Time Setup

### 1. Install Kaggle CLI + SDK

```bash
pip install kaggle kagglesdk
```

### 2. Authenticate

Set in `.env`:
```
KAGGLE_USERNAME=shakil19
KAGGLE_KEY=KGAT_xxxxxxxxxxxxx
```

Or set the environment variable for CLI use:
```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxxx"
```

### 3. Create the Kaggle Dataset (first time only)

```bash
cd kaggle-dataset
kaggle datasets create -p . --no-public
```

Expected output:
```
Successfully created dataset shakil19/athletic-intelligence-dataset
```

Or use the Python API:
```python
from src.backend.services.kaggle_manager import KaggleManager
km = KaggleManager(...)
km.create_dataset_if_missing()
```

### 4. Upload YOLO Model Weights (optional but speeds up kernel)

```bash
cp yolo11x-pose.pt kaggle-dataset/models/
kaggle datasets version -p kaggle-dataset -m "Add YOLO model weights"
```

If model is NOT in the dataset, `enable_internet: true` must be set in
`kernel-metadata.json` so the kernel can download from Ultralytics.

### 5. Create the Kaggle Kernel (first push creates it)

```bash
$env:KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxxx"
kaggle kernels push -p kaggle-kernel --accelerator NvidiaTeslaT4
```

Expected output:
```
Kernel version 1 successfully created.
URL: https://www.kaggle.com/shakil19/athletic-intelligence-kernel
```

---

## Per-Job Flow (automated by backend)

```python
# 1. Dataset update — adds video + config for this job
success, msg = km.push_job_to_dataset(
    job_id='abc123',
    video_path='/path/to/match.mp4',
    frame_number=450,
    incident_type='offside',
)

# 2. Kernel push — triggers new T4 GPU run
success, kernel_ref = km.push_kernel()

# 3. Poll status until COMPLETE or ERROR
while True:
    status, failure_msg = km.get_kernel_status()
    if status == 'complete':
        break
    elif status == 'failed':
        raise RuntimeError(failure_msg)
    time.sleep(5)

# 4. Download results from /kaggle/working/
success, msg = km.download_results('./outputs/abc123/')
```

---

## SDK Method Reference

| Method | SDK Call | Return |
|--------|----------|--------|
| `push_job_to_dataset()` | `api.dataset_create_version(folder, version_notes, dir_mode='skip')` | `(bool, str)` |
| `create_dataset_if_missing()` | `api.dataset_create_new(folder, public=False)` | `(bool, str)` |
| `push_kernel()` | `api.kernels_push(folder, acc='NvidiaTeslaT4')` | `(bool, str)` |
| `get_kernel_status()` | `api.kernels_status('{user}/{slug}')` | `(str, str)` |
| `download_results()` | `api.kernels_output(kernel, path, force=True)` | `(bool, str)` |
| `get_kernel_logs()` | `api.kernels_logs(kernel)` | `str` |

### KernelWorkerStatus Enum Values

| Value | Int | Meaning |
|-------|-----|---------|
| QUEUED | 0 | Waiting for GPU allocation |
| RUNNING | 1 | Currently executing |
| COMPLETE | 2 | Finished successfully |
| ERROR | 3 | Failed with error |
| CANCEL_REQUESTED | 4 | Cancel in progress |
| CANCEL_ACKNOWLEDGED | 5 | Cancelled |
| NEW_SCRIPT | 6 | Not yet started |

---

## Dataset Structure

```
kaggle-dataset/                  ← local staging dir
├── dataset-metadata.json        ← Kaggle dataset config
├── input/
│   ├── current_job.json         ← pointer to current job (overwritten each run)
│   ├── {job_id}.mp4             ← video file (one at a time)
│   └── {job_id}_config.json     ← incident parameters
├── models/
│   └── yolo11x-pose.pt          ← pre-cached YOLO weights (optional)
└── src/
    └── kernel/                  ← source code (kaggle_main.py imports from here)
        ├── config.py
        ├── main.py
        └── modules/
```

**On Kaggle (after mount):**
```
/kaggle/input/athletic-intelligence-dataset/
├── input/current_job.json
├── input/{job_id}.mp4
└── src/kernel/...

/kaggle/working/                 ← ONLY this dir is persisted and downloadable
└── {job_id}/
    ├── verdict.json
    ├── clip_offside.mp4
    ├── freeze_offside.png
    └── diagram_offside.png
```

---

## Kernel Structure

```
kaggle-kernel/
├── kernel-metadata.json    ← Kaggle kernel config
└── kaggle_main.py          ← entry point (imports from dataset src/)
```

### kernel-metadata.json

```json
{
  "id": "shakil19/athletic-intelligence-kernel",
  "code_file": "kaggle_main.py",
  "kernel_type": "script",
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["shakil19/athletic-intelligence-dataset"]
}
```

**Note:** GPU accelerator type is set via the CLI flag `--accelerator NvidiaTeslaT4`,
NOT in the JSON (setting it in JSON doesn't work — confirmed from LingGym experience).

---

## Key Lessons from LingGym Production

1. **Dataset mount path varies** — search both:
   - `/kaggle/input/athletic-intelligence-dataset`
   - `/kaggle/input/datasets/shakil19/athletic-intelligence-dataset`
   → `kaggle_main.py` searches both paths

2. **Only `/kaggle/working` is retrievable** — NOT `/kaggle/output`
   → All results written to `/kaggle/working/{job_id}/`

3. **Accelerator must be set via CLI flag** — `--accelerator NvidiaTeslaT4`
   → `kernels_push(folder, acc='NvidiaTeslaT4')`

4. **`enable_internet: false` is fine** — model weights are pre-cached in dataset
   → Set `enable_internet: true` only if model is NOT pre-cached

5. **`dir_mode='skip'` in dataset_create_version** — does not overwrite existing files,
   only adds new ones. Perfect for incremental video uploads.

---

## Monitoring

```bash
# Check kernel status
kaggle kernels status shakil19/athletic-intelligence-kernel

# Continuous polling (PowerShell)
do {
    $s = kaggle kernels status shakil19/athletic-intelligence-kernel 2>&1
    Write-Host "$(Get-Date -Format HH:mm:ss) $s"
    if ($s -match 'complete|error') { break }
    Start-Sleep 20
} while ($true)

# View logs
kaggle kernels output shakil19/athletic-intelligence-kernel -p ./debug/
cat debug/__results__.html
```

---

## Switching LOCAL_PROCESSING ↔ Kaggle

In `.env`:
```bash
# Local mode (default) — runs kernel.main directly, no Kaggle needed
LOCAL_PROCESSING=true

# Kaggle mode — full Kaggle dataset + kernel pipeline
LOCAL_PROCESSING=false
```

Local mode uses `asyncio.run_in_executor` to call `process_single_incident()`
without blocking the event loop. Falls back to simulated verdict if ML deps
(torch, ultralytics) are not installed.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `404 Not Found` on kernels_status | Kernel slug wrong | Check URL at kaggle.com/code/shakil19/ |
| Dataset not mounted | Wrong dataset_sources in metadata | Verify `"shakil19/athletic-intelligence-dataset"` |
| Results not downloadable | Written to /kaggle/output not /kaggle/working | Kernel writes to `/kaggle/working/` only |
| `CUDA error: no kernel image` | P100 GPU selected | Force T4 via `acc='NvidiaTeslaT4'` |
| Kernel stuck at QUEUED | GPU queue wait | Normal, up to 5 minutes |
| `enable_internet: false` fails | Model not cached | Add model to dataset OR set internet=true |
