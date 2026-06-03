#!/usr/bin/env python3
"""
Development launcher for Atlético Intelligence backend.

Usage:
    python run.py              # Start API server (default port 8000)
    python run.py --port 9000  # Custom port
    python run.py --kaggle     # Run Kaggle kernel locally (batch mode)
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies() -> bool:
    """Verify required packages are installed."""
    missing = []
    checks = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
    }
    for module, package in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: uv pip install {' '.join(missing)}")
        return False

    print("All dependencies OK")
    return True


def download_model() -> None:
    """Download YOLO11x-pose model weights (first run only)."""
    from ultralytics import YOLO
    model_file = PROJECT_ROOT / 'yolo11x-pose.pt'
    if not model_file.exists():
        print("Downloading yolo11x-pose model weights (first run only)...")
        YOLO('yolo11x-pose.pt')  # Ultralytics auto-downloads to current dir
        print("Model ready.")
    else:
        print(f"Model found: {model_file}")


def run_api_server(host: str = '0.0.0.0', port: int = 8000) -> None:
    """Start FastAPI backend server."""
    import uvicorn
    print(f"\nStarting API server at http://{host}:{port}")
    print(f"API docs: http://localhost:{port}/docs")
    print(f"Frontend: http://localhost:{port}/static/index.html\n")
    uvicorn.run(
        'src.backend.main:app',
        host=host,
        port=port,
        reload=True,
        log_level='info',
    )


def run_kaggle_kernel() -> None:
    """Run Kaggle kernel processing locally (batch mode, no server)."""
    from src.kernel.main import main as kernel_main
    print("\nRunning Kaggle kernel in local batch mode...")
    print(f"Input dir: {PROJECT_ROOT / 'input'}")
    print(f"Output dir: {PROJECT_ROOT / 'output'}\n")
    kernel_main()


def main() -> None:
    parser = argparse.ArgumentParser(description='Atlético Intelligence launcher')
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=8000, help='API server port')
    parser.add_argument('--kaggle', action='store_true', help='Run kernel locally in batch mode')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency checks')
    args = parser.parse_args()

    print('=' * 55)
    print('  Atlético Intelligence — Offside & Goal Detection')
    print('=' * 55)

    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        download_model()

    if args.kaggle:
        run_kaggle_kernel()
    else:
        run_api_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
