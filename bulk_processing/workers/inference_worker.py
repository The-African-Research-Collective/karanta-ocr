from celery import current_task
from utils.job_manager import JobManager
from .vllm_client import VLLMClient
from .celery_app import celery_app


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3})
def process_request(self, job_id: str, task_data: dict):
    """Process a single inference request"""

    task_id = task_data["task_id"]
    request_data = task_data["request"]

    # Initialize job manager (get output dir from task or config)
    job_manager = JobManager("./output")  # or get from config

    try:
        # Update status to processing
        job_manager.update_task_status(job_id, task_id, "processing")

        # Get VLLM client for this worker's GPU
        worker_name = current_task.request.hostname
        gpu_id = extract_gpu_id_from_worker(worker_name)
        vllm_client = VLLMClient(gpu_id)

        # Make inference request
        result = vllm_client.generate(
            prompt=request_data["prompt"],
            model=request_data.get("model", "default"),
            max_tokens=request_data.get("max_tokens", 100),
            temperature=request_data.get("temperature", 0.7),
        )

        # Update status to completed
        job_manager.update_task_status(job_id, task_id, "completed", result=result)

        return {"task_id": task_id, "status": "completed", "result": result}

    except Exception as e:
        error_msg = str(e)

        # Update status to failed
        job_manager.update_task_status(job_id, task_id, "failed", error=error_msg)

        # Retry logic is handled by Celery decorator
        raise self.retry(countdown=60, exc=e)


def extract_gpu_id_from_worker(worker_name: str) -> int:
    """Extract GPU ID from worker name (e.g., 'worker_gpu_0@hostname')"""
    if "gpu_" in worker_name:
        return int(worker_name.split("gpu_")[1].split("@")[0])
    return 0
