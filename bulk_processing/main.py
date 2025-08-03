#!/usr/bin/env python3
import os
import uuid
import time
import argparse
import jsonlines

from utils.job_manager import JobManager
from utils.gpu_router import GPURouter
from workers.celery_app import celery_app
from tqdm import tqdm


def process_batch_job(job, job_manager, db_path, output_path, model_name, ports=None):
    """Process batch job with failure recovery"""
    router = GPURouter(ports=ports)

    print(f"Processing job {job['job_id']}: {job['stats']['total']} requests")
    print(f"Progress: {job['stats']['completed']}/{job['stats']['total']} completed")

    # Get pending tasks
    pending_tasks = job_manager.get_pending_tasks(job["job_id"])

    if not pending_tasks:
        print("All tasks completed!")
        return

    # Submit pending tasks to Celery
    for i, task in tqdm(enumerate(pending_tasks)):
        queue = router.get_best_queue(task.get("model", "default"))

        celery_app.send_task(
            "workers.inference_worker.process_request",
            args=[job["job_id"], task, db_path, output_path, model_name],
            queue=queue,
            task_id=task["task_id"],
        )

        if (i + 1) % 50 == 0:
            time.sleep(300)

    print(f"Submitted {len(pending_tasks)} tasks to queues")
    print("Monitor progress with: celery -A workers.celery_app flower")
    print(f"Results will be saved to: {job['config']['output_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file with requests")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--job-id", help="Resume existing job")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per task"
    )
    parser.add_argument(
        "--ports", nargs="*", type=int, default=None, help="List of GPU ports to use"
    )
    parser.add_argument(
        "--model-name", type=str, help="Model name to use for inference"
    )

    args = parser.parse_args()

    # Initialize job manager
    # Create db in the output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    db_path = os.path.join(args.output, "batch_jobs.db")
    job_manager = JobManager(args.output, db_path=db_path)

    if args.job_id:
        # Resume existing job
        print(f"Resuming job: {args.job_id}")
        job = job_manager.load_job(args.job_id)
    elif os.path.exists(os.path.join(args.output, "job_id.txt")):
        # Load job ID from file
        with open(os.path.join(args.output, "job_id.txt"), "r") as f:
            job_id = f.read().strip()

        print(f"Resuming job: {job_id}")
        job = job_manager.load_job(job_id)
    else:
        # Create new job
        job_id = str(uuid.uuid4())

        # write job_id to a file in the output directory
        job_file = os.path.join(args.output, "job_id.txt")
        with open(job_file, "w") as f:
            f.write(job_id)

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
    process_batch_job(
        job, job_manager, db_path, args.output, args.model_name, args.ports
    )


if __name__ == "__main__":
    main()
