from celery import Celery

celery_app = Celery("batch_inference")

celery_app.config_from_object(
    {
        "broker_url": "redis://localhost:6379/0",
        "result_backend": "redis://localhost:6379/0",
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "task_routes": {
            "inference_worker.process_request": {"queue": "default"},
            "worker.inference_worker.process_request": {"queue": "default"},
        },
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "worker_max_tasks_per_child": 100,
    }
)

# Import tasks
from workers.inference_worker import process_request  # unused import
