"""
Kaggle dataset and kernel management using the official Kaggle Python SDK.

Workflow per job:
  1. push_job_to_dataset()  — copy video + config JSON to staging dir, call
                              dataset_create_version() to add them to Kaggle
  2. push_kernel()          — kernels_push() triggers a new kernel run on T4
  3. get_kernel_status()    — poll kernels_status() until COMPLETE or ERROR
  4. download_results()     — kernels_output() fetches /kaggle/working/ contents
"""

import json
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus


# Map SDK enum int values to human-readable strings
_STATUS_MAP: Dict[int, str] = {
    KernelWorkerStatus.QUEUED.value: 'queued',
    KernelWorkerStatus.RUNNING.value: 'running',
    KernelWorkerStatus.COMPLETE.value: 'complete',
    KernelWorkerStatus.ERROR.value: 'failed',
    KernelWorkerStatus.CANCEL_REQUESTED.value: 'running',
    KernelWorkerStatus.CANCEL_ACKNOWLEDGED.value: 'failed',
    KernelWorkerStatus.NEW_SCRIPT.value: 'queued',
}

# Accelerator identifier for T4 GPU
_T4_ACCELERATOR = 'NvidiaTeslaT4'


class KaggleManager:
    """Kaggle API wrapper for dataset uploads and kernel execution."""

    def __init__(
        self,
        dataset_name: str,
        kernel_slug: str,
        username: str,
        dataset_staging_dir: str,
        kernel_dir: str,
    ):
        """Initialize and authenticate with Kaggle.

        Args:
            dataset_name: Kaggle dataset slug (e.g. 'athletic-intelligence-dataset')
            kernel_slug:  Kaggle kernel slug (e.g. 'athletic-intelligence-kernel')
            username:     Kaggle username
            dataset_staging_dir: Local dir holding dataset-metadata.json and input/
            kernel_dir:   Local dir holding kernel-metadata.json and kaggle_main.py
        """
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.username = username
        self.dataset_ref = f'{username}/{dataset_name}'
        self.kernel_ref = f'{username}/{kernel_slug}'
        self.dataset_staging_dir = Path(dataset_staging_dir)
        self.kernel_dir = Path(kernel_dir)

        self.api = KaggleApi()
        self.api.authenticate()

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def push_job_to_dataset(
        self,
        job_id: str,
        video_path: str,
        frame_number: int,
        incident_type: str,
    ) -> Tuple[bool, str]:
        """Stage video + config and push a new dataset version.

        Copies the video and a config JSON to the local staging dir, then
        calls dataset_create_version() so the kernel can read them.

        Args:
            job_id:        Unique job identifier
            video_path:    Local path to the uploaded video file
            frame_number:  Incident frame index
            incident_type: 'offside' or 'goal'

        Returns:
            (success: bool, message: str)
        """
        try:
            video_src = Path(video_path)
            input_dir = self.dataset_staging_dir / 'input'
            input_dir.mkdir(parents=True, exist_ok=True)

            # Copy video with job_id prefix so kernel can find it
            video_dst = input_dir / f'{job_id}{video_src.suffix}'
            shutil.copy2(video_src, video_dst)

            # Write job config JSON
            config = {
                'job_id': job_id,
                'video_filename': video_dst.name,
                'incident_type': incident_type,
                'frame_number': frame_number,
                'submitted_at': datetime.utcnow().isoformat(),
            }
            config_path = input_dir / f'{job_id}_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Also write current_job.json as a stable pointer
            current_job_path = input_dir / 'current_job.json'
            with open(current_job_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Push new dataset version (dir_mode='skip' keeps existing files)
            self.api.dataset_create_version(
                folder=str(self.dataset_staging_dir),
                version_notes=f'Job {job_id} — {incident_type}',
                quiet=True,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode='skip',
            )

            return True, f'Dataset updated with job {job_id}'

        except Exception as e:
            return False, f'Dataset push failed: {e}'

    def create_dataset_if_missing(self) -> Tuple[bool, str]:
        """Create the Kaggle dataset if it does not exist yet.

        Call once during initial setup. Subsequent updates use push_job_to_dataset.

        Returns:
            (success: bool, message: str)
        """
        try:
            self.api.dataset_create_new(
                folder=str(self.dataset_staging_dir),
                public=False,
                quiet=True,
                convert_to_csv=False,
                dir_mode='skip',
            )
            return True, f'Dataset created: {self.dataset_ref}'
        except Exception as e:
            err = str(e)
            if '409' in err or 'already exists' in err.lower():
                return True, f'Dataset already exists: {self.dataset_ref}'
            return False, f'Dataset creation failed: {e}'

    # ------------------------------------------------------------------
    # Kernel operations
    # ------------------------------------------------------------------

    def push_kernel(self) -> Tuple[bool, str]:
        """Push the kernel to Kaggle, triggering a new run on T4 GPU.

        Reads kernel-metadata.json + kaggle_main.py from self.kernel_dir.

        Returns:
            (success: bool, message: str)
        """
        try:
            response = self.api.kernels_push(
                folder=str(self.kernel_dir),
                acc=_T4_ACCELERATOR,
            )
            ref = getattr(response, 'ref', self.kernel_ref)
            return True, str(ref)
        except Exception as e:
            return False, f'Kernel push failed: {e}'

    def get_kernel_status(self) -> Tuple[str, str]:
        """Poll the kernel's current execution status.

        Returns:
            (status: str, message: str)
            status values: 'queued' | 'running' | 'complete' | 'failed'
        """
        try:
            response = self.api.kernels_status(self.kernel_ref)
            status_enum_val = response.status.value
            status_str = _STATUS_MAP.get(status_enum_val, 'unknown')
            failure_msg = response.failure_message or ''
            return status_str, failure_msg
        except Exception as e:
            return 'unknown', f'Status check failed: {e}'

    def download_results(self, local_output_dir: str) -> Tuple[bool, str]:
        """Download /kaggle/working/ outputs to local_output_dir.

        Args:
            local_output_dir: Where to save downloaded files

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
            return True, f'Downloaded {len(files)} file(s) to {output_path}'
        except Exception as e:
            return False, f'Download failed: {e}'

    def get_kernel_logs(self) -> str:
        """Fetch kernel execution logs (useful for debugging failures).

        Returns:
            Log string or error message
        """
        try:
            return self.api.kernels_logs(self.kernel_ref)
        except Exception as e:
            return f'Could not fetch logs: {e}'
