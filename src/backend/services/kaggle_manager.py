"""
Kaggle dataset and kernel management using the official Kaggle Python SDK.

Automated per-job workflow (no manual steps):
  1. _sync_source_to_staging()   — copy src/kernel/ into dataset staging dir
  2. _clean_old_input_files()    — remove prior job's video to keep staging lean
  3. push_job_to_dataset()       — stage video + config, push dataset version
                                   (auto-creates dataset on first run)
  4. push_kernel()               — kernels_push(acc='NvidiaTeslaT4'), triggers run
  5. get_kernel_status()         — poll until COMPLETE or ERROR
  6. download_results()          — kernels_output() fetches /kaggle/working/
"""

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

log = logging.getLogger('kaggle_manager')

_STATUS_MAP: Dict[int, str] = {
    KernelWorkerStatus.QUEUED.value: 'queued',
    KernelWorkerStatus.RUNNING.value: 'running',
    KernelWorkerStatus.COMPLETE.value: 'complete',
    KernelWorkerStatus.ERROR.value: 'failed',
    KernelWorkerStatus.CANCEL_REQUESTED.value: 'running',
    KernelWorkerStatus.CANCEL_ACKNOWLEDGED.value: 'failed',
    KernelWorkerStatus.NEW_SCRIPT.value: 'queued',
}

_T4_ACCELERATOR = 'NvidiaTeslaT4'

# Source code directory relative to project root
_SRC_KERNEL_RELPATH = Path('src') / 'kernel'


class KaggleManager:
    """Fully automated Kaggle API wrapper — no manual steps after init."""

    def __init__(
        self,
        dataset_name: str,
        kernel_slug: str,
        username: str,
        dataset_staging_dir: str,
        kernel_dir: str,
        project_root: Optional[str] = None,
    ):
        """Initialize and authenticate.

        Args:
            dataset_name:        Kaggle dataset slug
            kernel_slug:         Kaggle kernel slug
            username:            Kaggle username
            dataset_staging_dir: Local dir with dataset-metadata.json
            kernel_dir:          Local dir with kernel-metadata.json + kaggle_main.py
            project_root:        Project root for syncing src/kernel/ (auto-detected)
        """
        from kaggle.api.kaggle_api_extended import KaggleApi
        import os

        self.username = username
        self.dataset_ref = f'{username}/{dataset_name}'
        self.kernel_ref = f'{username}/{kernel_slug}'
        self.dataset_staging_dir = Path(dataset_staging_dir)
        self.kernel_dir = Path(kernel_dir)
        self.project_root = Path(project_root) if project_root else Path(kernel_dir).parents[1]

        # SDK v2 auth routing:
        # - KGAT_xxx tokens → KAGGLE_API_TOKEN (OAuth access token path)
        # - Legacy 32-char hex keys → KAGGLE_USERNAME + KAGGLE_KEY
        key = os.getenv('KAGGLE_KEY', '')
        if key.startswith('KGAT_'):
            os.environ['KAGGLE_API_TOKEN'] = key
            log.debug('KGAT token detected — set KAGGLE_API_TOKEN for SDK v2 auth')

        self.api = KaggleApi()
        self.api.authenticate()

        log.info('KaggleManager authenticated as %s', username)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_source_to_staging(self) -> None:
        """Copy src/kernel/ into the staging dir so the Kaggle kernel can import it.

        kaggle_main.py does:  sys.path.insert(0, dataset_root)
                              from src.kernel.main import process_single_incident
        So we need staging_dir/src/kernel/ to mirror the local src/kernel/.
        """
        src = self.project_root / _SRC_KERNEL_RELPATH
        dst = self.dataset_staging_dir / _SRC_KERNEL_RELPATH

        if not src.exists():
            log.warning('src/kernel not found at %s — skipping sync', src)
            return

        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        # Ensure package __init__.py exists at src/ level too
        src_init = self.dataset_staging_dir / 'src' / '__init__.py'
        src_init.parent.mkdir(parents=True, exist_ok=True)
        src_init.touch()

        log.info('Synced src/kernel/ → staging (%d files)', sum(1 for _ in dst.rglob('*')))

    def _clean_old_input_files(self) -> None:
        """Remove prior job's video from the staging root dir.

        Files are kept flat at staging root (no subdirs).
        current_job.json is overwritten on each push, not deleted.
        """
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        for f in self.dataset_staging_dir.iterdir():
            if f.is_file() and f.suffix.lower() in video_exts:
                f.unlink()
                log.debug('Removed old video from staging: %s', f.name)

    def _ensure_dataset_exists(self) -> None:
        """Create the Kaggle dataset (first run). Ignored if already exists."""
        try:
            self.api.dataset_create_new(
                folder=str(self.dataset_staging_dir),
                public=False,
                quiet=True,
                convert_to_csv=False,
                dir_mode='overwrite',
            )
            log.info('Dataset created: %s', self.dataset_ref)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ('409', 'already exists', 'Conflict')):
                log.debug('Dataset already exists: %s', self.dataset_ref)
            else:
                raise RuntimeError(f'Dataset creation failed: {e}') from e

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def push_job_to_dataset(
        self,
        job_id: str,
        video_path: str,
        incident_type: str,
        frame_number: Optional[int],
    ) -> Tuple[bool, str]:
        """Stage video + config and push a new dataset version.

        Fully automated:
          1. Sync src/kernel/ source code into staging
          2. Clean prior job's video from staging
          3. Copy this job's video + write config JSONs
          4. Push dataset version (auto-creates dataset on first run)

        Args:
            job_id:        Unique job identifier
            video_path:    Local path to uploaded video
            incident_type: 'offside' or 'goal'
            frame_number:  Frame to analyze (None → kernel picks middle frame)

        Returns:
            (success: bool, message: str)
        """
        try:
            # Step 1: Clean prior input files (video only)
            self._clean_old_input_files()

            # Step 3: Stage files flat at staging root (no subdirectories)
            # Kaggle's dataset_create_new/version skip subdirs by default.
            # Putting files at root avoids the issue entirely.
            video_src = Path(video_path)
            video_dst = self.dataset_staging_dir / f'{job_id}{video_src.suffix}'
            shutil.copy2(video_src, video_dst)
            log.info('[%s] Staged video: %s', job_id, video_dst.name)

            job_config = {
                'job_id': job_id,
                'video_filename': video_dst.name,
                'incident_type': incident_type,
                'frame_number': frame_number,
                'submitted_at': datetime.utcnow().isoformat(),
            }

            # Per-job config (stable filename so kernel always reads current job)
            current_job_file = self.dataset_staging_dir / 'current_job.json'
            with open(current_job_file, 'w') as f:
                json.dump(job_config, f, indent=2)

            # Step 4: Push dataset version
            log.info('[%s] Pushing dataset version to Kaggle', job_id)
            try:
                self.api.dataset_create_version(
                    folder=str(self.dataset_staging_dir),
                    version_notes=f'Job {job_id[:8]} - {incident_type}',
                    quiet=True,
                    convert_to_csv=False,
                    delete_old_versions=False,
                    dir_mode='skip',   # flat files only; no subdirs in staging root
                )
            except Exception as e:
                err = str(e)
                if any(code in err for code in ('403', '404', 'not found', 'Forbidden')):
                    log.info('[%s] Dataset does not exist, creating it first...', job_id)
                    self._ensure_dataset_exists()
                    # dataset_create_new uploads all root-level files including video
                else:
                    raise

            log.info('[%s] Dataset push complete', job_id)
            return True, f'Dataset updated: job {job_id[:8]}'

        except Exception as e:
            log.error('[%s] Dataset push failed: %s', job_id, e)
            return False, f'Dataset push failed: {e}'

    # ------------------------------------------------------------------
    # Kernel operations
    # ------------------------------------------------------------------

    def push_kernel(self, job_id: str = '') -> Tuple[bool, str]:
        """Push kernel to Kaggle, triggering a new T4 GPU run.

        Args:
            job_id: For logging only

        Returns:
            (success: bool, kernel_ref_or_error: str)
        """
        try:
            log.info('[%s] Pushing kernel to Kaggle (T4 GPU)', job_id)
            response = self.api.kernels_push(
                folder=str(self.kernel_dir),
                acc=_T4_ACCELERATOR,
            )
            # response.ref is '/code/owner/slug' — strip the URL prefix
            raw_ref = getattr(response, 'ref', '') or ''
            ref = raw_ref.removeprefix('/code/') or self.kernel_ref
            log.info('[%s] Kernel pushed: %s', job_id, ref)
            return True, str(ref)
        except Exception as e:
            log.error('[%s] Kernel push failed: %s', job_id, e)
            return False, f'Kernel push failed: {e}'

    def get_kernel_status(self, kernel_ref: Optional[str] = None) -> Tuple[str, str]:
        """Poll kernel execution status.

        Args:
            kernel_ref: Override the default kernel ref (e.g. from push response)

        Returns:
            (status: str, failure_message: str)
            status: 'queued' | 'running' | 'complete' | 'failed' | 'unknown'
        """
        ref = kernel_ref or self.kernel_ref
        try:
            response = self.api.kernels_status(ref)
            status_str = _STATUS_MAP.get(response.status.value, 'unknown')
            return status_str, response.failure_message or ''
        except Exception as e:
            return 'unknown', f'Status check error: {e}'

    def download_results(self, local_output_dir: str) -> Tuple[bool, str]:
        """Download /kaggle/working/ to local_output_dir.

        Args:
            local_output_dir: Destination directory

        Returns:
            (success: bool, message: str)
        """
        try:
            output_path = Path(local_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            files = []
            try:
                files, _ = self.api.kernels_output(
                    kernel=self.kernel_ref,
                    path=str(output_path),
                    force=True,
                    quiet=True,
                )
            except UnicodeEncodeError:
                # Windows CP1252: SDK progress print fails but files still download.
                # Scan the output dir for what was actually saved.
                files = [str(p) for p in output_path.rglob('*') if p.is_file()]
                log.warning('Encoding error in SDK output, but %d file(s) saved', len(files))
            log.info('Downloaded %d file(s) to %s', len(files), output_path)
            return True, f'Downloaded {len(files)} file(s)'
        except Exception as e:
            return False, f'Download failed: {e}'

    def get_kernel_logs(self) -> str:
        """Fetch execution logs for debugging failures."""
        try:
            return self.api.kernels_logs(self.kernel_ref)
        except Exception as e:
            return f'Could not fetch logs: {e}'
