# utils/job_manager.py
from .database import JobDatabase
from pathlib import Path
import json
from datetime import datetime


class JobManager:
    def __init__(self, output_dir: str, db_path: str = "batch_jobs.db"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.db = JobDatabase(db_path)

    def create_job(self, job_id: str, requests: list, config: dict):
        return self.db.create_job(job_id, requests, config)

    def load_job(self, job_id: str):
        job = self.db.get_job(job_id)
        if not job:
            raise FileNotFoundError(f"Job {job_id} not found")
        return job

    def update_task_status(
        self, job_id: str, task_id: str, status: str, result=None, error=None
    ):
        success = self.db.update_task_status(job_id, task_id, status, result, error)

        # Save individual result file for completed tasks
        if success and status == "completed" and result is not None:
            result_file = self.results_dir / f"{task_id}.json"
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "task_id": task_id,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )

    def get_pending_tasks(self, job_id: str):
        job = self.load_job(job_id)
        max_retries = job["config"].get("max_retries", 3)
        return self.db.get_pending_tasks(job_id, max_retries)
