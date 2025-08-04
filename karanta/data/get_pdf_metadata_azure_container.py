"""
This script processes PDF files in an Azure blob container,
converts them to images, and generates JSONL files for TogetherAI's batch inference.
It then asynchronously submits JSONL files to TogetherAI as soon as they are ready.

Sample usage:
     python -m karanta.data.get_pdf_metadata --container_name mycontainer --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""

import argparse
import asyncio
from asyncio import Semaphore
import logging
import os
from pathlib import Path

from azure.core.pipeline.transport import AioHttpTransport
from azure.storage.blob.aio import ContainerClient


from dotenv import load_dotenv
from together import Together
import yaml

from karanta.data.utils import process_pdf, batch_submitter, jsonl_writer

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "pdf_metadata.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)

load_dotenv()
transport = AioHttpTransport(read_timeout=600)

############################ AZURE BLOB STORAGE CONFIGURATION ############################
AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
##########################################################################################

client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
PROMPT_PATH = "configs/prompts/classification_batch_inference.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Using TogetherAI's batch inference return information on the language contained in PDFs in a blob container."
    )
    parser.add_argument(
        "--container_name",
        help="Container name in Azure blob containing PDFs of interest",
    )
    parser.add_argument(
        "--output", default="output", help="Base output name to store output files"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="Preferred TogetherAI model for classification. See available models at https://docs.together.ai/docs/batch-inference#model-availability",
    )
    parser.add_argument(
        "--max_concurrent_pdfs",
        type=int,
        default=1,
        help="Maximum number of concurrent PDF processing tasks (default 1)",
    )
    return parser.parse_args()


async def process_blob_pdf(
    container: ContainerClient,
    blob_name: str,
    model: str,
    prompt_template: dict,
    sem: Semaphore,
    queue: asyncio.Queue,
):
    async with sem:
        try:
            # Download PDF to memory
            stream = await container.download_blob(blob_name)
            pdf_file = await stream.readall()

            logging.info(f"Processing blob: {blob_name}")

            # Convert to JSONL line
            json_line = process_pdf(pdf_file, blob_name, model, prompt_template)

            # Send line to writer queue
            await queue.put(json_line)
            logging.info(f"Successfully processed {blob_name}")

        except Exception as e:
            logging.error(f"Failed to process {blob_name}: {e}")


async def concurrent_stream_process(
    container_name: str,
    model: str,
    max_concurrent_pdfs: int,
    output: str,
    prompt_template: dict,
):
    sem = Semaphore(max_concurrent_pdfs)
    queue = asyncio.Queue()
    completed_files_queue = asyncio.Queue()

    container = ContainerClient(
        account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
        container_name=container_name,
        credential=AZURE_SAS_TOKEN,
        transport=transport,
    )

    writer_task = asyncio.create_task(
        jsonl_writer(queue, completed_files_queue, output, 100)
    )
    submitter_task = asyncio.create_task(batch_submitter(client, completed_files_queue))

    async with container:
        tasks = []
        async for blob in container.list_blobs():
            if blob.name.lower().endswith(".pdf"):
                tasks.append(
                    asyncio.create_task(
                        process_blob_pdf(
                            container, blob.name, model, prompt_template, sem, queue
                        )
                    )
                )

        logging.info(f"Found {len(tasks)} PDF files in container {container_name}")

        # Wait for all processing to finish
        await asyncio.gather(*tasks)

    # Signal writer to stop
    await queue.put(None)
    await writer_task

    # Signal batch submitter to stop
    await completed_files_queue.put(None)
    await submitter_task

    logging.info("All processing completed successfully.")


def main():
    args = parse_args()
    # Load Yaml template
    with open(PROMPT_PATH, "r") as f:
        template = yaml.safe_load(f)
    asyncio.run(
        concurrent_stream_process(
            container_name=args.container_name,
            model=args.model,
            max_concurrent_pdfs=args.max_concurrent_pdfs,
            output=args.output,
            prompt_template=template,
        )
    )


if __name__ == "__main__":
    main()
