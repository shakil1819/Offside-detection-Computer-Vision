# Atlético Intelligence Backend API

FastAPI backend for offside and goal detection processing.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or using uv:
uv pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` to add:
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key
- `KAGGLE_DATASET_NAME`: Name of your Kaggle dataset

### 3. Run Backend Server

```bash
# Development mode (auto-reload)
python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
uvicorn src.backend.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs` (Swagger UI)

## Architecture

### Directory Structure

```
src/backend/
├── main.py                    # FastAPI app
├── config.py                  # Configuration
├── models.py                  # Pydantic schemas
├── services/
│   ├── job_manager.py         # Job state management
│   └── kaggle_manager.py      # Kaggle integration
├── utils/
│   └── validators.py          # Input validation
├── data/
│   ├── uploads/               # Temp uploaded videos
│   └── outputs/               # Processing results
└── storage/
    └── jobs.json              # Job state persistence
```

### Services

#### JobManager
Manages job state with in-memory store + JSON file persistence.

```python
from src.backend.services.job_manager import JobManager

job_manager = JobManager(state_file=Path('jobs.json'))

# Create job
job_id = job_manager.create_job('video.mp4')

# Update job
job_manager.update_job(job_id, status='processing', progress=50)

# Get job
job = job_manager.get_job(job_id)
```

#### KaggleManager
Wraps Kaggle API for dataset and kernel management.

```python
from src.backend.services.kaggle_manager import KaggleManager

kaggle = KaggleManager(
    dataset_name='athletic-intelligence-dataset',
    username='your_username',
    key='your_key'
)

# Upload file
success, msg = kaggle.upload_file('local.mp4', 'input/video.mp4')

# Trigger kernel
success, kernel_id = kaggle.trigger_kernel(job_id, config)

# Check status
status, msg = kaggle.get_kernel_status(kernel_id)
```

## API Endpoints

### Upload Video
```
POST /upload
Content-Type: multipart/form-data

Body:
  file: <video_file>

Response:
  {
    "job_id": "abc123-def456",
    "status": "uploading",
    "message": "Video received, queued for processing"
  }
```

### Check Job Status
```
GET /status/{job_id}

Response:
  {
    "job_id": "abc123",
    "status": "processing",
    "progress": 45,
    "verdict": null,
    "confidence": null,
    "error": null,
    "created_at": "2024-06-04T10:30:00",
    "started_at": "2024-06-04T10:30:05",
    "completed_at": null
  }
```

### Download Results
```
GET /download/{job_id}?format=video

Formats:
  - video: MP4 clip (default)
  - json: Verdict JSON
  - zip: All results
```

### List Jobs
```
GET /jobs?limit=20

Response:
  {
    "count": 5,
    "jobs": [
      {
        "job_id": "abc123",
        "video_filename": "match_001.mp4",
        "status": "completed",
        "progress": 100,
        "verdict": "OFFSIDE",
        "confidence": 0.92,
        "created_at": "2024-06-04T10:30:00"
      }
    ]
  }
```

### Delete Job
```
DELETE /jobs/{job_id}

Response:
  {
    "message": "Job deleted: abc123"
  }
```

## Job State Machine

```
┌─────────────────────────────────────────────────────────┐
│                    JOB LIFECYCLE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User uploads video                                     │
│           │                                             │
│           ├─→ [PENDING] Job created, queued            │
│           │                                             │
│           ├─→ [UPLOADING] Uploading to Kaggle          │
│           │                                             │
│           ├─→ [PROCESSING] Kernel running               │
│           │    - Clip extraction (0-20%)                │
│           │    - YOLO inference (20-80%)                │
│           │    - Diagram generation (80-95%)            │
│           │    - Result upload (95-100%)                │
│           │                                             │
│           ├─→ [COMPLETED] Results ready                │
│           │    - verdict: OFFSIDE/ONSIDE/GOAL/NO-GOAL   │
│           │    - confidence: 0-1.0                       │
│           │    - clip_path: MP4 file                     │
│           │                                             │
│           └─→ [FAILED] Error occurred                   │
│                - error: Error message                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black src/ tests/
ruff check src/ tests/
```

### Hot Reload
The server automatically reloads when code changes:

```bash
uvicorn src.backend.main:app --reload
```

## Deployment

### Production

```bash
# Run with gunicorn (4 workers)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.backend.main:app

# Or use uvicorn directly
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t athletic-intelligence-api .
docker run -p 8000:8000 -e KAGGLE_USERNAME=... -e KAGGLE_KEY=... athletic-intelligence-api
```

## Monitoring

### Logs
Check server output for real-time logs:
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Job History
View jobs.json for persistent state:
```bash
cat src/backend/storage/jobs.json | python -m json.tool
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Kaggle API Not Configured
If `KAGGLE_USERNAME` or `KAGGLE_KEY` not set:
- Backend still works, but cannot push to Kaggle
- Jobs are simulated locally
- Useful for testing UI without Kaggle setup

### Video Too Large
- Max size: 500 MB
- Check file size before upload
- Compress video: `ffmpeg -i input.mp4 -crf 28 output.mp4`

### Job Stuck in Processing
- Check job status with `GET /status/{job_id}`
- Default timeout: 5 minutes (300 seconds)
- After timeout, job marked as failed

## Next Steps

1. **Kaggle Kernel Setup**
   - Create dataset on Kaggle
   - Upload YOLO model to dataset
   - Create Kaggle kernel with processing code

2. **Frontend Integration**
   - Serve `src/frontend/index.html` from backend
   - Or run on separate port with CORS enabled

3. **Production Deployment**
   - Use Docker or serverless platform
   - Set up monitoring and alerting
   - Configure Kaggle credentials securely

## API Documentation

Full API documentation available at `/docs` when running:

```bash
uvicorn src.backend.main:app --reload
```

Then visit: http://localhost:8000/docs

## Contributing

Follow CLAUDE.md protocol for code changes.
