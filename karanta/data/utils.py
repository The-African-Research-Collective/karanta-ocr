import os
import json
import logging
import time
import io
import base64
from enum import Enum
import asyncio
from typing import List, Optional, Union, Set
from pathlib import Path
from functools import wraps
from pydantic import BaseModel

from pdf2image import convert_from_path, convert_from_bytes
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class DocumentCategory(str, Enum):
    """Categories used in document sorting"""

    FICTION = "fiction"
    NON_FICTION = "non_fiction"
    ACADEMIC_REFERENCE = "academic_reference"
    RELIGIOUS = "religious"
    MAGAZINES_PERIODICALS = "magazines_periodicals"
    NEWSPAPER = "newspaper"


class DocumentClassification(BaseModel):
    """Structured output format for document classification"""

    category: DocumentCategory
    language: str
    confidence: float


def convert_pdf2image(data_path: Path, output_dir: Path):
    """Convert PDF files to images using pdf2image."""
    return convert_from_path(
        data_path, output_folder=output_dir, last_page=3, fmt="jpg"
    )


def convert_pdf2image_from_bytes(pdf_file: bytes):
    """Convert PDF files to images using pdf2image."""
    return convert_from_bytes(pdf_file, last_page=3, fmt="jpg")


def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_mixed_datasets(
    dataset_sources: Union[dict, list],
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
    shuffle_data: bool = True,
    save_dir: Optional[str] = None,
    ensure_columns: Optional[List[str]] = None,
    include_source_column: bool = False,
) -> DatasetDict:
    """
    Prepare and mix datasets from multiple sources, controlling the size or percentage each dataset contributes.
    """
    splits = splits or ["train", "test"]
    configs = configs or [None] * len(dataset_sources)
    required_columns = required_columns or []

    if len(configs) != len(dataset_sources):
        raise ValueError(
            "The number of configs must match the number of dataset sources."
        )

    mixed_datasets = DatasetDict()
    train_datasets = []
    test_datasets = []

    for (source, fraction_or_count), config in zip(dataset_sources.items(), configs):
        for split in splits:
            try:
                dataset = load_dataset(source, config, split=split)
            except Exception:
                dataset = load_from_disk(os.path.join(source, split))

            if shuffle_data:
                dataset = dataset.shuffle(seed=42)

            if ensure_columns:
                missing_columns = [
                    col for col in ensure_columns if col not in dataset.column_names
                ]
                if missing_columns:
                    raise ValueError(
                        f"Missing required columns: {missing_columns} in dataset {source}."
                    )

            if include_source_column:
                dataset = dataset.add_column("source", [source] * len(dataset))

            # Control the size or percentage of the dataset
            if isinstance(fraction_or_count, float) and 0 < fraction_or_count <= 1:
                dataset = dataset.select(range(int(len(dataset) * fraction_or_count)))
            elif isinstance(fraction_or_count, int) and fraction_or_count > 0:
                dataset = dataset.select(range(min(fraction_or_count, len(dataset))))

            if split == "train":
                train_datasets.append(dataset)
            elif split == "test":
                test_datasets.append(dataset)

    if train_datasets:
        mixed_datasets["train"] = concatenate_datasets(train_datasets)
    if test_datasets:
        mixed_datasets["test"] = concatenate_datasets(test_datasets)

    if save_dir:
        for split, dataset in mixed_datasets.items():
            dataset.save_to_disk(os.path.join(save_dir, f"{split}_mixed"))

    return mixed_datasets


@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def upload_metadata_to_hub(
    metadata: dict, filename: str, repo_id: str, repo_dir: str
) -> None:
    """Upload metadata to the Hugging Face Hub."""
    with open("temp_metadata.json", "w") as f:
        json.dump(metadata, f)
    api = HfApi()
    api.upload_file(
        path_or_fileobj="temp_metadata.json",
        path_in_repo=f"{repo_dir}/{filename}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    os.remove("temp_metadata.json")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def push_folder_to_hub(folder: str, repo_id: str, branch: Optional[str] = None) -> None:
    """Push a folder to the Hugging Face Hub."""
    api = HfApi()
    if not api.repo_exists(repo_id):
        api.create_repo(repo_id, exist_ok=True)
    if branch:
        api.create_branch(repo_id, branch, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        revision=branch,
        commit_message="Upload folder",
    )
    logger.info(
        f"Pushed folder to https://huggingface.co/{repo_id}/tree/{branch or 'main'}"
    )


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def process_pdf(pdf_input, pdf_name: str, model: str, prompt_template: dict) -> str:
    """Build TogetherAI JSONL request from PDF."""
    schema = DocumentClassification.model_json_schema()
    base_prompt = prompt_template["base_prompt"]

    base_data = {
        "custom_id": pdf_name,
        "body": {
            "model": model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": base_prompt}]}
            ],
            "response_format": {
                "type": "json_schema",
                "schema": schema,
            },
        },
    }

    # Convert PDF to images
    images = convert_pdf2image_from_bytes(pdf_input)

    # Encode up to 3 images
    image_content = []
    for image in images[:3]:
        encoded_image = encode_image(image)
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    base_data["body"]["messages"][0]["content"].extend(image_content)

    return json.dumps(base_data)


async def batch_call(client, batch_file_path: str):
    """Submits a JSONL file to TogetherAI's batch API and waits for completion."""
    try:
        file_resp = client.files.upload(file=batch_file_path, purpose="batch-api")
        batch = client.batches.create_batch(
            file_resp.id, endpoint="/v1/chat/completions"
        )
        base_name = os.path.splitext(batch_file_path)[0]
        output_file = f"{base_name}_results.jsonl"

        logger.info(f"Batch submitted: {batch_file_path}")

        while True:
            batch_status = client.batches.get_batch(batch.id)
            if batch_status.status == "COMPLETED":
                client.files.retrieve_content(
                    id=batch_status.output_file_id, output=output_file
                )
                logger.info(f"Batch completed: {batch_file_path} -> {output_file}")

                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            result = json.loads(line)
                            file_name = result.get("custom_id", "unknown")
                            logger.info(f"{file_name}: success")
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse result line in {output_file}"
                            )
                break

            elif batch_status.status == "FAILED":
                client.files.retrieve_content(
                    id=batch_status.error_file_id, output=output_file
                )
                logger.error(
                    f"Batch failed for {batch_file_path}: {batch_status.error}"
                )
                raise ValueError(f"Batch failed: {batch_status.error}")

            await asyncio.sleep(10)

    except Exception as e:
        logger.exception(f"Error submitting batch {batch_file_path}: {str(e)}")


async def batch_submitter(client, completed_files_queue: asyncio.Queue):
    """Listens for completed JSONL files and submits them as batches."""
    submitted_files: Set[str] = set()

    while True:
        file_path = await completed_files_queue.get()
        if file_path is None:
            break

        if file_path in submitted_files:
            continue

        logger.info(f"Submitting {file_path} to TogetherAI batch API")
        submitted_files.add(file_path)
        await batch_call(client, file_path)
        logger.info(f"Completed batch submission for {file_path}")


async def jsonl_writer(
    queue: asyncio.Queue,
    completed_files_queue: asyncio.Queue,
    base_output_name: str,
    max_file_size_mb: int = 100,
):
    """Consumes JSON lines from queue and writes to rotated JSONL files."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    max_file_size = max_file_size_mb * 1024 * 1024
    file_index = 1
    current_file_size = 0
    output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
    f = open(output_file, "w", encoding="utf-8")
    logger.info(f"Writing to {output_file}")

    while True:
        json_line = await queue.get()
        if json_line is None:
            break

        line_bytes = (json_line + "\n").encode("utf-8")
        line_size = len(line_bytes)

        # Rotate file if needed
        if current_file_size + line_size > max_file_size:
            f.close()
            await completed_files_queue.put(str(output_file))

            file_index += 1
            output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
            f = open(output_file, "w", encoding="utf-8")
            logger.info(f"Rotating to {output_file}")
            current_file_size = 0

        f.write(json_line + "\n")
        current_file_size += line_size

    f.close()
    await completed_files_queue.put(str(output_file))
    logger.info(f"Completed writing {file_index} JSONL file(s).")


def openai_response_format_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "page_response",
            "schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "additionalProperties": False,
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
            "strict": True,
        },
    }
