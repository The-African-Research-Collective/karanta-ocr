"""
This scipts procesess PDF files in specified folder, converts them to images and uses a model to classify them based on language and general category.
It generates a JSONL file which it uses with together batch inference.

Sample usage:
     python -m karanta.data.get_pdf_metadata --folder_path /path/to/pdf/folder --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""

import argparse
import asyncio
from asyncio import Semaphore
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

from together import Together

from karanta.data.utils import process_pdf, jsonl_writer, batch_submitter

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "pdf_metadata.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)

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
        help="TogetherAI model for classification. See available models at https://docs.together.ai/docs/batch-inference#model-availability",
    )
    parser.add_argument(
        "--max_concurrent_pdfs",
        type=int,
        default=1,
        help="Maximum concurrent PDF processing (default=1)",
    )
    return parser.parse_args()


async def process_local_pdf(
    pdf_path: Path,
    model: str,
    prompt_template: dict,
    sem: Semaphore,
    queue: asyncio.Queue,
):
    async with sem:
        try:
            with open(pdf_path, "rb") as f:
                pdf_input = f.read()
            pdf_name = os.path.basename(pdf_path)

            json_line = process_pdf(pdf_input, pdf_name, model, prompt_template)
            await queue.put(json_line)

            logging.info(f"Processed: {pdf_path}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_path}: {e}")


async def concurrent_folder_process(
    folder_path: str,
    model: str,
    max_concurrent_pdfs: int,
    output: str,
    prompt_template: dict,
):
    sem = Semaphore(max_concurrent_pdfs)
    queue = asyncio.Queue()
    completed_files_queue = asyncio.Queue()

    writer_task = asyncio.create_task(
        jsonl_writer(queue, completed_files_queue, output, 100)
    )
    submitter_task = asyncio.create_task(batch_submitter(client, completed_files_queue))

    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDF files found in folder: {folder_path}")
        return

    tasks = [
        asyncio.create_task(process_local_pdf(pdf, model, prompt_template, sem, queue))
        for pdf in pdf_files
    ]
    await asyncio.gather(*tasks)

    await queue.put(None)
    await writer_task

    await completed_files_queue.put(None)
    await submitter_task

    logging.info("PDFs processing and batch submissions complete.")


def main():
    args = parse_args()
    # Load YAML template
    with open(PROMPT_PATH, "r") as f:
        template = yaml.safe_load(f)
    asyncio.run(
        concurrent_folder_process(
            folder_path=args.folder_path,
            model=args.model,
            max_concurrent_pdfs=args.max_concurrent_pdfs,
            output=args.output,
            prompt_template=template,
        )
    )


if __name__ == "__main__":
    main()
