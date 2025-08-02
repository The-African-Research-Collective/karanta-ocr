import logging
from celery import current_task
from workers.vllm_client import get_vllm_client_for_worker
from workers.celery_app import celery_app
from utils.job_manager import JobManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global job manager instance
# This will be initialized when the worker starts
_job_manager = None


def get_job_manager(output_dir, db_path) -> JobManager:
    """
    Get or create job manager instance

    Returns:
        JobManager instance
    """
    global _job_manager

    if _job_manager is None:
        _job_manager = JobManager(output_dir, db_path)
        logger.info(
            f"Initialized job manager with output_dir='{output_dir}', db_path='{db_path}'"
        )

    return _job_manager


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3})
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

    # Get job manager and VLLM client
    job_manager = get_job_manager(output_path, db_path)

    try:
        # Update status to processing
        job_manager.update_task_status(job_id, task_id, "processing")

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

        # Update job status with result
        job_manager.update_task_status(job_id, task_id, "completed", result=result)

        return result

    except Exception as e:
        job_manager.update_task_status(job_id, task_id, "failed", error=str(e))
        raise self.retry(countdown=60, exc=e)
