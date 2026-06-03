"""Job state management with in-memory store + JSON file persistence."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class Job:
    """Job state record."""

    job_id: str
    video_filename: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    kaggle_kernel_id: Optional[str] = None
    kaggle_dataset_path: Optional[str] = None
    progress: int = 0
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    result_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary with datetime serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create from dictionary with datetime parsing."""
        data = data.copy()
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class JobManager:
    """In-memory job manager with JSON file persistence."""

    def __init__(self, state_file: Path):
        """Initialize job manager.

        Args:
            state_file: Path to JSON file for persistence
        """
        self.state_file = Path(state_file)
        self.jobs: Dict[str, Job] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load job state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for job_id, job_data in data.items():
                        self.jobs[job_id] = Job.from_dict(job_data)
            except Exception as e:
                print(f"Warning: Could not load jobs from {self.state_file}: {e}")
                self.jobs = {}

    def _save_to_file(self) -> None:
        """Save job state to JSON file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save jobs to {self.state_file}: {e}")

    def create_job(self, video_filename: str) -> str:
        """Create a new job.

        Args:
            video_filename: Name of the uploaded video

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            video_filename=video_filename,
            status='pending',
            created_at=datetime.utcnow(),
        )
        self.jobs[job_id] = job
        self._save_to_file()
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update

        Returns:
            True if successful, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        self._save_to_file()
        return True

    def list_jobs(self, limit: int = 100) -> list:
        """List all jobs (most recent first).

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of Job objects
        """
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return sorted_jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete job.

        Args:
            job_id: Job identifier

        Returns:
            True if successful, False if job not found
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_to_file()
            return True
        return False

    def cleanup_old_jobs(self, age_hours: int = 24) -> int:
        """Remove jobs older than specified hours.

        Args:
            age_hours: Remove jobs older than this many hours

        Returns:
            Number of jobs removed
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=age_hours)
        old_jobs = [
            job_id for job_id, job in self.jobs.items()
            if job.created_at < cutoff
        ]

        for job_id in old_jobs:
            del self.jobs[job_id]

        if old_jobs:
            self._save_to_file()

        return len(old_jobs)
