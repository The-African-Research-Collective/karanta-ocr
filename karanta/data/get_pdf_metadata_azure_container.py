"""
This script processes PDF files in an Azure blob container,
converts them to images, and generates JSONL files for TogetherAI's batch inference.
It then asynchronously submits JSONL files to TogetherAI as soon as they are ready.

Sample usage:
     python -m karanta.data.get_pdf_metadata --container_name mycontainer --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""

import argparse
import logging
import os
from pathlib import Path

from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from together import Together
import yaml

from karanta.data.utils import (
    process_pdf_to_batch_line,
    jsonl_writer,
    batch_call,
)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "pdf_metadata.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()
AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")

PROMPT_PATH = "configs/prompts/classification_batch_inference.yaml"
client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process PDFs from an Azure Blob container and submit to TogetherAI batch API for language classification."
    )
    parser.add_argument(
        "--container_name", required=True, help="Azure blob container name"
    )
    parser.add_argument(
        "--output", default="output", help="Base output name for JSONL files"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="TogetherAI model for classification",
    )
    return parser.parse_args()


def process_pdf_blob(
    container: ContainerClient, blob_name: str, model: str, prompt_template: dict
) -> str:
    """Download and process a single PDF blob to a JSONL line suitable for batch inference.

    Args:
        container (ContainerClient): Azure Blob container client.
        blob_name (str): Name of the PDF blob to process.
        model (str): TogetherAI model name for classification.
        prompt_template (dict): Prompt template for batch inference.

    Returns:
        str: JSONL-formatted line.
    """
    try:
        blob_client = container.get_blob_client(blob_name)
        pdf_file = blob_client.download_blob().readall()
        json_line = process_pdf_to_batch_line(
            pdf_file, blob_name, model, prompt_template
        )
        logger.info(f"Processed blob: {blob_name}")
        return json_line
    except Exception as e:
        logger.error(f"Failed to process blob {blob_name}: {e}")
        return None


def classify_pdfs_in_container(
    container_name: str, model: str, output: str, prompt_template: dict
):
    """Process all PDF files in an Azure Blob container.

    Args:
        container_name (str): Azure Blob container with PDF files.
        model (str): TogetherAI model for batch inference.
        output (str): Base name for JSONL output files.
        prompt_template (dict): Prompt template for batch inference.
    """
    account_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
    container = ContainerClient(account_url, container_name, credential=AZURE_SAS_TOKEN)

    logger.info(f"Listing blobs in container: {container_name}")
    blob_list = container.list_blobs()

    json_lines = []
    count = 0
    for blob in blob_list:
        if blob.name.lower().endswith(".pdf"):
            json_line = process_pdf_blob(container, blob.name, model, prompt_template)
            if json_line:
                json_lines.append(json_line)
                count += 1

    logger.info(f"Processed {count} PDF(s) from container {container_name}")

    if not json_lines:
        logger.warning("No PDF files processed. Exiting.")
        return

    completed_files = jsonl_writer(json_lines, output)

    for file_path in completed_files:
        batch_call(client, file_path)

    logger.info("All PDF processing and batch submissions complete.")


def main():
    """CLI entry point."""
    args = parse_args()

    with open(PROMPT_PATH, "r") as f:
        template = yaml.safe_load(f)

    classify_pdfs_in_container(
        container_name=args.container_name,
        model=args.model,
        output=args.output,
        prompt_template=template,
    )


if __name__ == "__main__":
    main()
