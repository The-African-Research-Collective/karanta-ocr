# utils/job_manager.py
from .database import JobDatabase
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """
        Direct task status update (for non-batch operations)
        For batch operations, use BatchProcessor instead
        """
        success = self.db.update_task_status(job_id, task_id, status, result, error)

        logger.info(
            f"Updated task {task_id} status to '{status}' for job {job_id}. Success: {success}"
        )

        # Save individual result file for completed tasks
        if success and status == "completed" and result is not None:
            self._write_result_file(task_id, result)

        return success

    def _write_result_file(self, task_id: str, result):
        """Write individual result file"""
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
        max_retries = job["config"].get("max_retries", 10)
        return self.db.get_pending_tasks(job_id, max_retries)

    def get_job_analytics(self, job_id: str):
        """Get detailed analytics for a job"""
        return self.db.get_job_analytics(job_id)

    def batch_update_tasks(self, updates: list):
        """
        Batch update multiple tasks across potentially multiple jobs

        Args:
            updates: List of dicts with keys: job_id, task_id, status, result, error
        """
        from collections import defaultdict

        # Group by job_id for efficient processing
        job_updates = defaultdict(list)
        for update in updates:
            job_updates[update["job_id"]].append(update)

        # Process each job's updates in a single transaction
        for job_id, job_update_list in job_updates.items():
            with self.db.get_connection() as conn:
                try:
                    for update in job_update_list:
                        self._execute_task_update(conn, update)

                    # Update job stats once for all changes
                    self._update_job_statistics(conn, job_id)

                    conn.commit()
                    logger.info(
                        f"Batch updated {len(job_update_list)} tasks for job {job_id}"
                    )

                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error in batch update for job {job_id}: {e}")
                    raise

    def _execute_task_update(self, conn, update: dict):
        """Execute a single task update within an existing transaction"""
        task_id = update["task_id"]
        status = update["status"]
        result = update.get("result")
        error = update.get("error")

        # Build update query dynamically (same logic as original)
        set_clauses = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]

        if status == "processing":
            set_clauses.append("started_at = CURRENT_TIMESTAMP")
        elif status == "completed":
            set_clauses.extend(
                [
                    "completed_at = CURRENT_TIMESTAMP",
                    "processing_time_ms = (julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400000",
                ]
            )
            if result is not None:
                set_clauses.append("result_data = ?")
                params.append(json.dumps(result))
        elif status == "failed":
            set_clauses.append("attempts = attempts + 1")
            if error:
                set_clauses.append("error_message = ?")
                params.append(error)

        params.append(task_id)

        conn.execute(
            f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = ?", params
        )

    def _update_job_statistics(self, conn, job_id: str):
        """Update job statistics within an existing transaction"""
        conn.execute(
            """
            UPDATE jobs SET
                completed_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'completed'),
                failed_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'failed'),
                processing_tasks = (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'processing'),
                updated_at = CURRENT_TIMESTAMP,
                status = CASE 
                    WHEN (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status IN ('completed', 'failed')) = total_tasks
                    THEN 'completed'
                    WHEN (SELECT COUNT(*) FROM tasks WHERE job_id = ? AND status = 'processing') > 0
                    THEN 'processing'
                    ELSE 'created'
                END
            WHERE job_id = ?
        """,
            (job_id, job_id, job_id, job_id, job_id, job_id),
        )
