# Atlético Intelligence Frontend

Vanilla HTML/CSS/JavaScript web interface for offside and goal detection.

## Features

- ✅ Video upload with drag-and-drop
- ✅ Real-time progress tracking
- ✅ Verdict display (OFFSIDE/ONSIDE/GOAL/NO-GOAL)
- ✅ Embedded video player for results
- ✅ Job history with recent analyses
- ✅ Responsive design (mobile-friendly)
- ✅ Zero build step (vanilla JS)

## Quick Start

### 1. Serve Frontend

Option A: With backend (recommended)
```bash
cd src/backend
uvicorn main:app --reload
# Frontend auto-served at http://localhost:8000/
```

Option B: Static file server
```bash
cd src/frontend
python -m http.server 8080
# Visit http://localhost:8080
```

Option C: Live server (VS Code extension)
- Open `src/frontend/index.html`
- Right-click → "Open with Live Server"

### 2. Configure API Endpoint

Edit `src/frontend/js/api.js`:
```javascript
const api = new APIClient('http://localhost:8000');
```

Or leave blank to auto-detect:
```javascript
const api = new APIClient();  // Uses current origin
```

## Architecture

### File Structure

```
src/frontend/
├── index.html                 # Single HTML page
├── css/
│   └── style.css             # Responsive styling
├── js/
│   ├── api.js                # API client
│   ├── uploader.js           # File upload handler
│   ├── status-poller.js      # Job polling
│   └── app.js                # Main orchestrator
└── README.md
```

### Components

#### HTMLElements
- Upload section: File input with drag-drop
- Progress section: Animated progress bar
- Results section: Verdict badge + video player
- Error section: Error messages
- History section: Recent jobs

#### JavaScript Classes

**APIClient** (`api.js`)
- Wrapper around fetch API
- Handles requests to backend
- Methods: uploadVideo, getStatus, downloadResults, listJobs

**Uploader** (`uploader.js`)
- Manages file input and validation
- Drag-and-drop support
- File size/format validation
- Updates display after selection

**StatusPoller** (`status-poller.js`)
- Polls `/status/{job_id}` endpoint
- Exponential backoff on errors
- Callbacks: onProgress, onComplete, onError
- Configurable polling interval

**App** (`app.js`)
- Main orchestrator
- Manages UI state transitions
- Handles user interactions
- Displays results and history

## User Flow

```
┌─────────────────────────────────────────┐
│  1. User selects/drags video file      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. Frontend validates file              │
│     - Format (MP4, AVI, MOV, MKV)       │
│     - Size (max 500 MB)                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. Upload to backend                   │
│     POST /upload                        │
│     ← job_id                           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. Poll for status every 2 seconds     │
│     GET /status/{job_id}               │
│     - Shows progress bar               │
│     - Updates percentage               │
└────────────────┬────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
   SUCCESS              FAILED
      │                     │
      ▼                     ▼
┌──────────────┐    ┌──────────────┐
│ Display:     │    │ Show error   │
│ - Verdict    │    │ message      │
│ - Confidence │    │ Try Again btn│
│ - Video      │    └──────────────┘
│ - Download   │
└──────────────┘
```

## Configuration

### Backend API URL

Default: Current origin (`http://localhost:8000`)

Override in `app.js`:
```javascript
const api = new APIClient('http://api.example.com');
```

### Polling Intervals

Edit `status-poller.js`:
```javascript
const poller = new StatusPoller(
    2000,    // Initial: 2 seconds
    30000    // Max: 30 seconds
);
```

### UI Customization

Edit `css/style.css`:
- Primary color: `--primary-color: #1e3a8a`
- Secondary color: `--secondary-color: #06b6d4`
- Font: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto...`

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

**Not supported:**
- IE 11 (uses modern JavaScript features)

## Accessibility

- Semantic HTML structure
- ARIA labels on buttons
- Keyboard navigation support
- Focus states on all interactive elements
- Color contrast WCAG AA compliant
- Reduced motion support

## Performance

- No build step (vanilla JS)
- CSS-only animations (smooth 60 FPS)
- Single-page app (no page reloads)
- Minimal JavaScript (~15 KB)
- ~100 KB total bundle size

## Development

### Local Testing

```bash
# Start backend
cd src/backend
uvicorn main:app --reload

# Start frontend (in another terminal)
cd src/frontend
python -m http.server 8080
```

Visit: http://localhost:8080

### Debug Mode

Add to `app.js`:
```javascript
window.DEBUG = true;
console.log('Job:', this.currentJobId);
```

Open browser DevTools (F12) to see logs.

### Offline Testing

Simulate backend responses in `api.js`:
```javascript
async uploadVideo(file) {
    console.log('Mock: Upload', file.name);
    return { job_id: 'mock-123', status: 'uploading' };
}
```

## Deployment

### On Backend Server

The frontend is auto-served by FastAPI at `/`:

```bash
# Backend serves frontend
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

Visit: http://your-server:8000

### On Static Server

```bash
# Copy frontend to web server
scp -r src/frontend/* user@example.com:/var/www/athletic-intelligence/

# Configure API endpoint in index.html or js/api.js
```

### On CDN

```bash
# Upload frontend to CloudFlare, Vercel, Netlify, etc.
# Configure CORS on backend to allow requests
```

### Docker

```dockerfile
FROM node:18-alpine AS build
# Optional: build/optimize

FROM python:3.10-slim
COPY src/frontend/ /app/frontend/
COPY src/backend/ /app/backend/
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0"]
```

## Troubleshooting

### "Cannot connect to backend"
- Check backend is running: `curl http://localhost:8000/health`
- Check API URL in `js/api.js`
- Check CORS configuration in backend

### Video player not playing
- Check browser console for errors (F12)
- Ensure video format is supported
- Check `video` MIME type in backend response

### Progress bar stuck
- Check job status: `/status/{job_id}`
- Check backend logs for errors
- Timeout: job marked failed after 5 minutes

### File upload fails
- Max file size: 500 MB
- Supported formats: MP4, AVI, MOV, MKV
- Check browser file picker permissions

## Contributing

Follow guidelines:
1. Use vanilla JS (no frameworks)
2. Keep CSS responsive (mobile-first)
3. Maintain accessibility standards
4. Comment non-obvious code
5. Test in multiple browsers

## License

MIT License - see LICENSE file
