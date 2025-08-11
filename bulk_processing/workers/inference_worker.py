# inference_worker.py
import logging
import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Any
from celery import current_task
from workers.vllm_client import get_vllm_client_for_worker
from workers.celery_app import celery_app
from utils.job_manager import JobManager
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for job managers and batch processors
_job_managers = {}
_batch_processors = {}
_manager_lock = threading.Lock()


class BatchProcessor:
    """Handles batching of database writes and file operations for a specific job"""

    def __init__(
        self,
        job_manager: JobManager,
        job_id: str,
        batch_size: int = 50,
        flush_interval: float = 10,
    ):
        self.job_manager = job_manager
        self.job_id = job_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Batch queues
        self.db_updates = deque()
        self.file_writes = deque()

        # Threading
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.flush_thread = None

        self.start_flush_thread()

    def start_flush_thread(self):
        """Start background thread for periodic flushing"""
        if self.flush_thread is None or not self.flush_thread.is_alive():
            self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self.flush_thread.start()
            logger.info(f"Started batch processor flush thread for job {self.job_id}")

    def _flush_loop(self):
        """Background thread that flushes batches periodically"""
        while not self.stop_event.wait(self.flush_interval):
            try:
                self.flush_all()
            except Exception as e:
                logger.error(f"Error in flush loop for job {self.job_id}: {e}")

    def queue_db_update(
        self, task_id: str, status: str, result: Any = None, error: str = None
    ):
        """Queue a database update for batch processing"""
        update_data = {
            "job_id": self.job_id,
            "task_id": task_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": time.time(),
        }

        with self.lock:
            self.db_updates.append(update_data)

            # Force flush if batch is full
            if len(self.db_updates) >= self.batch_size:
                self._flush_db_updates_unsafe()

    def queue_file_write(self, task_id: str, result: Any):
        """Queue a file write for batch processing"""
        file_data = {
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with self.lock:
            self.file_writes.append(file_data)

            # Force flush if batch is full
            if len(self.file_writes) >= self.batch_size:
                self._flush_file_writes_unsafe()

    def _flush_db_updates_unsafe(self):
        """Flush database updates without acquiring lock (internal use)"""
        if not self.db_updates:
            return

        updates_to_process = list(self.db_updates)
        self.db_updates.clear()

        try:
            # All updates are for the same job, so process them together
            self._batch_update_job_tasks(updates_to_process)
            logger.info(
                f"Flushed {len(updates_to_process)} database updates for job {self.job_id}"
            )

        except Exception as e:
            logger.error(f"Error flushing database updates for job {self.job_id}: {e}")
            # Re-queue failed updates (simple retry strategy)
            with self.lock:
                self.db_updates.extendleft(reversed(updates_to_process))

    def _batch_update_job_tasks(self, updates: List[Dict]):
        """Update multiple tasks for this job in a single transaction"""
        with self.job_manager.db.get_connection() as conn:
            try:
                # Process all task updates in one transaction
                for update in updates:
                    self._execute_single_update(conn, update)

                # Update job statistics once for all changes
                self._update_job_stats(conn)

                conn.commit()
                logger.debug(
                    f"Batch updated {len(updates)} tasks for job {self.job_id}"
                )

            except Exception as e:
                conn.rollback()
                logger.error(f"Error in batch update for job {self.job_id}: {e}")
                raise

    def _execute_single_update(self, conn, update: Dict):
        """Execute a single task update within a transaction"""
        task_id = update["task_id"]
        status = update["status"]
        result = update["result"]
        error = update["error"]

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

    def _update_job_stats(self, conn):
        """Update job statistics and status"""
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
            (
                self.job_id,
                self.job_id,
                self.job_id,
                self.job_id,
                self.job_id,
                self.job_id,
            ),
        )

    def _flush_file_writes_unsafe(self):
        """Flush file writes without acquiring lock (internal use)"""
        if not self.file_writes:
            return

        files_to_write = list(self.file_writes)
        self.file_writes.clear()

        try:
            # Write all files in batch
            results_dir = self.job_manager.results_dir

            for file_data in files_to_write:
                result_file = results_dir / f"{file_data['task_id']}.json"
                with open(result_file, "w") as f:
                    json.dump(
                        {
                            "task_id": file_data["task_id"],
                            "result": file_data["result"],
                            "timestamp": file_data["timestamp"],
                        },
                        f,
                        indent=2,
                    )

            logger.info(
                f"Flushed {len(files_to_write)} file writes for job {self.job_id}"
            )

        except Exception as e:
            logger.error(f"Error flushing file writes for job {self.job_id}: {e}")
            # Re-queue failed writes
            with self.lock:
                self.file_writes.extendleft(reversed(files_to_write))

    def flush_all(self):
        """Manually flush all pending operations"""
        with self.lock:
            self._flush_db_updates_unsafe()
            self._flush_file_writes_unsafe()

    def shutdown(self):
        """Shutdown the batch processor"""
        self.stop_event.set()
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)

        # Final flush
        self.flush_all()
        logger.info(f"Batch processor shutdown complete for job {self.job_id}")


def get_job_manager_and_processor(
    job_id: str, output_dir: str, db_path: str
) -> tuple[JobManager, BatchProcessor]:
    """
    Get or create job manager and batch processor for a specific job

    Returns:
        tuple: (JobManager, BatchProcessor) instances for the job
    """
    global _job_managers, _batch_processors

    # Use job-specific key for caching
    cache_key = f"{job_id}:{db_path}"

    with _manager_lock:
        # Check if we already have instances for this job
        if cache_key not in _job_managers:
            # Create new job manager for this specific job
            job_manager = JobManager(output_dir, db_path)
            batch_processor = BatchProcessor(job_manager, job_id)

            _job_managers[cache_key] = job_manager
            _batch_processors[cache_key] = batch_processor

            logger.info(f"Created job manager and batch processor for job {job_id}")
            logger.info(f"  Output dir: {output_dir}")
            logger.info(f"  Database: {db_path}")

        return _job_managers[cache_key], _batch_processors[cache_key]


@celery_app.task(
    bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 10}
)
def process_request(
    self,
    job_id: str,
    task_data: dict,
    db_path: str,
    output_path: str,
    model_name: str = "default",
):
    task_id = task_data["task_id"]
    request_data = task_data["request"]

    logger.info(f"Processing task {task_id} for job {job_id}")
    logger.debug(f"Using database: {db_path}")
    logger.debug(f"Using output path: {output_path}")

    # Get job-specific manager and batch processor
    job_manager, batch_processor = get_job_manager_and_processor(
        job_id, output_path, db_path
    )

    try:
        # Update status to processing (queued for batch update)
        batch_processor.queue_db_update(task_id, "processing")

        # Get VLLM client for this worker's port
        worker_name = current_task.request.hostname
        logger.debug(f"Worker name: {worker_name}")

        vllm_client = get_vllm_client_for_worker(worker_name)
        logger.debug(f"Using VLLM client: {vllm_client}")

        # Make inference request
        result = vllm_client.generate(
            messages=request_data["messages"],
            model=request_data.get("model", "default")
            if not model_name
            else model_name,
            max_tokens=request_data.get("max_tokens", 6000),
            temperature=request_data.get("temperature", 0.1),
            response_format=request_data.get("response_format", None),
        )

        # Queue both database update and file write for batch processing
        batch_processor.queue_db_update(task_id, "completed", result=result)
        batch_processor.queue_file_write(task_id, result)

        logger.info(f"Completed task {task_id} for job {job_id}")
        return result

    except Exception as e:
        # Queue failed status update
        batch_processor.queue_db_update(task_id, "failed", error=str(e))
        logger.error(f"Failed task {task_id} for job {job_id}: {e}")
        raise self.retry(countdown=60, exc=e)


# Celery worker shutdown hooks
def shutdown_all_batch_processors():
    """Shutdown hook for Celery worker - clean up all batch processors"""
    global _batch_processors

    logger.info("Shutting down all batch processors...")

    for cache_key, batch_processor in _batch_processors.items():
        try:
            batch_processor.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down batch processor {cache_key}: {e}")

    _batch_processors.clear()
    _job_managers.clear()

    logger.info("All batch processors shutdown complete")


# Register shutdown hook
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup any periodic tasks if needed"""
    pass


# Optional: Add a manual flush task for admin purposes
@celery_app.task
def flush_job_batch_operations(job_id: str, db_path: str):
    """Manual task to flush batch operations for a specific job"""
    cache_key = f"{job_id}:{db_path}"

    global _batch_processors
    if cache_key in _batch_processors:
        _batch_processors[cache_key].flush_all()
        return f"Flushed operations for job {job_id}"
    else:
        return f"No active batch processor found for job {job_id}"


@celery_app.task
def flush_all_batch_operations():
    """Manual task to flush all pending batch operations across all jobs"""
    global _batch_processors

    flushed_count = 0
    for cache_key, batch_processor in _batch_processors.items():
        batch_processor.flush_all()
        flushed_count += 1

    return f"Flushed operations for {flushed_count} jobs"
