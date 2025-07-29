#!/usr/bin/env python3
import uuid
import argparse
import jsonlines
from utils.job_manager import JobManager
from utils.gpu_router import GPURouter
from workers.celery_app import celery_app


def process_batch_job(job, job_manager, gpus=None):
    """Process batch job with failure recovery"""
    router = GPURouter(gpus=gpus)

    print(f"Processing job {job['job_id']}: {job['stats']['total']} requests")
    print(f"Progress: {job['stats']['completed']}/{job['stats']['total']} completed")

    # Get pending tasks
    pending_tasks = job_manager.get_pending_tasks(job["job_id"])

    if not pending_tasks:
        print("All tasks completed!")
        return

    # Submit pending tasks to Celery
    for task in pending_tasks:
        queue = router.get_best_queue(task.get("model", "default"))

        celery_app.send_task(
            "inference_worker.process_request",
            args=[job["job_id"], task],
            queue=queue,
            task_id=task["task_id"],
        )

    print(f"Submitted {len(pending_tasks)} tasks to queues")
    print("Monitor progress with: celery -A workers.celery_app flower")
    print(f"Results will be saved to: {job['config']['output_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file with requests")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--job-id", help="Resume existing job")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per task"
    )
    parser.add_argument(
        "--gpu-ids", nargs="*", type=int, default=None, help="List of GPU IDs to use"
    )

    args = parser.parse_args()

    # Initialize job manager
    job_manager = JobManager(args.output)

    if args.job_id:
        # Resume existing job
        print(f"Resuming job: {args.job_id}")
        job = job_manager.load_job(args.job_id)
    else:
        # Create new job
        job_id = str(uuid.uuid4())
        print(f"Starting new job: {job_id}")

        # Load requests from input file
        with jsonlines.open(args.input, "r") as reader:
            requests = [obj for obj in reader]

        job = job_manager.create_job(
            job_id=job_id,
            requests=requests,
            config={
                "batch_size": args.batch_size,
                "max_retries": args.max_retries,
                "output_dir": args.output,
            },
        )

    # Process the job
    process_batch_job(job, job_manager, args.gpu_ids)


if __name__ == "__main__":
    main()
