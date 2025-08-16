#!/usr/bin/env python3
"""
Main script to handle bulk processing of requests with Celery and GPU routing.

How to Use:
    python3 main.py --input {file_containing_requests}  \
            --output {output_directory_to_store_files} \
            --ports {list of ports running vllm e.g. 8005 8006} \
            --model-name allenai/olmOCR-7B-0725-FP8
"""

import os
import uuid
import time
import argparse
import jsonlines
from pathlib import Path

from utils.job_manager import JobManager
from utils.gpu_router import GPURouter
from workers.celery_app import celery_app
from tqdm import tqdm

CELERY_BATCH_SIZE = (
    100  # This is number of tasks to submit before pausing for few minutes
)
CELERY_PAUSE_TIME = 300  # Pause time in seconds


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
        queue = router.get_best_queue()

        celery_app.send_task(
            "workers.inference_worker.process_request",
            args=[job["job_id"], task, db_path, output_path, model_name],
            queue=queue,
            task_id=task["task_id"],
        )

        if (i + 1) % CELERY_BATCH_SIZE == 0:
            time.sleep(CELERY_PAUSE_TIME)

    print(f"Submitted {len(pending_tasks)} tasks to queues")
    print("Monitor progress with: celery -A workers.celery_app flower")
    print(f"Results will be saved to: {job['config']['output_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Directory")
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
    input_path = Path(args.input)

    if input_path.exists() and not input_path.is_file():
        for file in input_path.glob("*.jsonl"):
            if not file.is_file():
                raise ValueError(f"File {file} is not a valid JSONL file.")

            # Create output directory from input file name
            output_dir = os.path.join(args.output, file.stem)
            print(f"Processing file: {file}, output directory: {output_dir}")
            # Initialize job manager
            # Create db in the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            db_path = os.path.join(output_dir, "batch_jobs.db")
            job_manager = JobManager(output_dir, db_path=db_path)

            if args.job_id:
                # Resume existing job
                print(f"Resuming job: {args.job_id}")
                job = job_manager.load_job(args.job_id)
            elif os.path.exists(
                os.path.join(output_dir, "job_id.txt")
            ) and os.path.exists(os.path.join(output_dir, "batch_jobs.db")):
                # Load job ID from file
                with open(os.path.join(output_dir, "job_id.txt"), "r") as f:
                    job_id = f.read().strip()

                print(f"Resuming job: {job_id}")
                job = job_manager.load_job(job_id)
            else:
                # Create new job
                job_id = str(uuid.uuid4())

                # write job_id to a file in the output directory
                job_file = os.path.join(output_dir, "job_id.txt")
                with open(job_file, "w") as f:
                    f.write(job_id)

                print(f"Starting new job: {job_id}")

                # Load requests from input file
                with jsonlines.open(file, "r") as reader:
                    requests = [obj for obj in reader]

                job = job_manager.create_job(
                    job_id=job_id,
                    requests=requests,
                    config={
                        "batch_size": args.batch_size,
                        "max_retries": args.max_retries,
                        "output_dir": output_dir,
                    },
                )

            # Process the job
            process_batch_job(
                job, job_manager, db_path, output_dir, args.model_name, args.ports
            )
    else:
        raise ValueError(
            f"Input path {args.input} must be a directory containing requests."
        )


if __name__ == "__main__":
    main()
