"""
This scipts procesess PDF files in specified folder, converts them to images and uses a model to classify them based on language and general category.
It generates a JSONL file which it uses with together batch inference.

Sample usage:
     python -m karanta.data.get_pdf_metadata --folder_path /path/to/pdf/folder --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

from together import Together

from karanta.data.utils import (
    process_pdf_to_batch_line,
    jsonl_writer,  # <-- create a sync version of jsonl_writer
    batch_call,  # <-- create a sync version of batch_call
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
PROMPT_PATH = "configs/prompts/classification_batch_inference.yaml"
client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process PDFs in a local folder and submit to TogetherAI batch API for language classification."
    )
    parser.add_argument(
        "--folder_path", required=True, help="Path to folder containing PDFs"
    )
    parser.add_argument(
        "--output", default="output", help="Base name for JSONL output files"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="TogetherAI model for classification. See: https://docs.together.ai/docs/batch-inference#model-availability",
    )
    parser.add_argument(
        "--max_concurrent_pdfs",
        type=int,
        default=1,
        help="(Ignored in sync mode) Max concurrent PDF processing",
    )
    return parser.parse_args()


def process_local_pdf_sync(pdf_path: Path, model: str, prompt_template: dict) -> str:
    """
    Reads a PDF and returns the JSONL line for batch submission.
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_input = f.read()
        pdf_name = os.path.basename(pdf_path)
        json_line = process_pdf_to_batch_line(
            pdf_input, pdf_name, model, prompt_template
        )
        logger.info(f"Processed: {pdf_path}")
        return json_line
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return None


def classify_pdfs_in_folder(
    folder_path: str, model: str, output: str, prompt_template: dict
):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in folder: {folder_path}")
        return

    # Process PDFs sequentially
    json_lines = []
    for pdf in pdf_files:
        json_line = process_local_pdf_sync(pdf, model, prompt_template)
        if json_line:
            json_lines.append(json_line)

    # Write JSONL files (rotates if >100MB)
    completed_files = jsonl_writer(json_lines, output)

    # Submit each completed file to TogetherAI
    for file_path in completed_files:
        batch_call(client, file_path)

    logger.info("PDF processing and batch submissions complete.")


def main():
    args = parse_args()

    # Load YAML template
    with open(PROMPT_PATH, "r") as f:
        template = yaml.safe_load(f)

    classify_pdfs_in_folder(
        folder_path=args.folder_path,
        model=args.model,
        output=args.output,
        prompt_template=template,
    )


if __name__ == "__main__":
    main()
