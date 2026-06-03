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

        self.username = username
        self.dataset_ref = f'{username}/{dataset_name}'
        self.kernel_ref = f'{username}/{kernel_slug}'
        self.dataset_staging_dir = Path(dataset_staging_dir)
        self.kernel_dir = Path(kernel_dir)
        self.project_root = Path(project_root) if project_root else Path(kernel_dir).parents[1]

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
        """Remove prior job's video and config from staging input/ dir.

        Keeps dataset staging lean — only the current job's files.
        current_job.json is overwritten (not deleted) so the kernel always
        reads the latest job.
        """
        input_dir = self.dataset_staging_dir / 'input'
        if not input_dir.exists():
            return

        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        for f in list(input_dir.iterdir()):
            if f.suffix.lower() in video_exts:
                f.unlink()
                log.debug('Removed old video from staging: %s', f.name)
            elif f.name.endswith('_config.json') and f.name != 'current_job.json':
                f.unlink()
                log.debug('Removed old config from staging: %s', f.name)

    def _ensure_dataset_exists(self) -> None:
        """Create the Kaggle dataset if it doesn't exist yet (first run only)."""
        try:
            self.api.dataset_create_new(
                folder=str(self.dataset_staging_dir),
                public=False,
                quiet=True,
                convert_to_csv=False,
                dir_mode='skip',
            )
            log.info('Dataset created: %s', self.dataset_ref)
        except Exception as e:
            err = str(e)
            if '409' in err or 'already exists' in err.lower():
                log.debug('Dataset already exists: %s', self.dataset_ref)
            else:
                raise

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
            # Step 1: Sync source code
            log.info('[%s] Syncing src/kernel/ to dataset staging', job_id)
            self._sync_source_to_staging()

            # Step 2: Clean prior input files
            self._clean_old_input_files()

            # Step 3: Stage this job's files
            video_src = Path(video_path)
            input_dir = self.dataset_staging_dir / 'input'
            input_dir.mkdir(parents=True, exist_ok=True)

            video_dst = input_dir / f'{job_id}{video_src.suffix}'
            shutil.copy2(video_src, video_dst)
            log.info('[%s] Staged video: %s', job_id, video_dst.name)

            job_config = {
                'job_id': job_id,
                'video_filename': video_dst.name,
                'incident_type': incident_type,
                'frame_number': frame_number,   # None = kernel picks middle frame
                'submitted_at': datetime.utcnow().isoformat(),
            }

            # Per-job config file
            config_file = input_dir / f'{job_id}_config.json'
            with open(config_file, 'w') as f:
                json.dump(job_config, f, indent=2)

            # Stable pointer (overwrite='overwrite' ensures kernel always sees current job)
            current_job_file = input_dir / 'current_job.json'
            with open(current_job_file, 'w') as f:
                json.dump(job_config, f, indent=2)

            # Step 4: Push dataset version (dir_mode='overwrite' updates current_job.json)
            log.info('[%s] Pushing dataset version to Kaggle', job_id)
            try:
                self.api.dataset_create_version(
                    folder=str(self.dataset_staging_dir),
                    version_notes=f'Job {job_id[:8]} — {incident_type}',
                    quiet=True,
                    convert_to_csv=False,
                    delete_old_versions=False,
                    dir_mode='overwrite',   # Ensure current_job.json is always updated
                )
            except Exception as e:
                if '404' in str(e) or 'not found' in str(e).lower():
                    # Dataset doesn't exist yet — create then version
                    log.info('[%s] Dataset not found, creating...', job_id)
                    self._ensure_dataset_exists()
                    self.api.dataset_create_version(
                        folder=str(self.dataset_staging_dir),
                        version_notes=f'Job {job_id[:8]} — {incident_type} (initial)',
                        quiet=True,
                        convert_to_csv=False,
                        delete_old_versions=False,
                        dir_mode='overwrite',
                    )
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
            ref = getattr(response, 'ref', self.kernel_ref)
            log.info('[%s] Kernel pushed: %s', job_id, ref)
            return True, str(ref)
        except Exception as e:
            log.error('[%s] Kernel push failed: %s', job_id, e)
            return False, f'Kernel push failed: {e}'

    def get_kernel_status(self) -> Tuple[str, str]:
        """Poll kernel execution status.

        Returns:
            (status: str, failure_message: str)
            status: 'queued' | 'running' | 'complete' | 'failed' | 'unknown'
        """
        try:
            response = self.api.kernels_status(self.kernel_ref)
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

            files, _ = self.api.kernels_output(
                kernel=self.kernel_ref,
                path=str(output_path),
                force=True,
                quiet=True,
            )
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
