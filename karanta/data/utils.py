import os
import json
import logging
import time
import yaml
import base64

from pathlib import Path
from functools import wraps
from jinja2 import Template
from PIL import Image
from io import BytesIO
from typing import List, Optional, Literal, Union

from pdf2image import convert_from_path
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel, Field

from karanta.constants import TARGET_IMAGE_DIM, PROMPT_PATH
from karanta.prompts.anchor import get_anchor_text
from karanta.data.process_pdf_utils import render_pdf_to_base64png

logger = logging.getLogger(__name__)


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

    Args:
        dataset_sources (Union[dict, list]): Sources of datasets to mix. Can be a dictionary or list.
            If a dictionary, keys are dataset sources, and values are fractions or counts to control contribution.
            If a list, it alternates between dataset sources and their corresponding fractions or counts.
        splits (Optional[List[str]]): Dataset splits to load and mix (e.g., "train", "test").
        configs (Optional[List[str]]): Configurations for datasets, if applicable.
        required_columns (Optional[List[str]]): Columns to retain in the final dataset.
        shuffle_data (bool): Whether to shuffle the datasets.
        save_dir (Optional[str]): Directory to save the mixed dataset.
        ensure_columns (Optional[List[str]]): Columns that must exist in the dataset.
        include_source_column (bool): Whether to include a column indicating the source of the data.

    Returns:
        DatasetDict: A dictionary containing the mixed datasets.
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
    """
    Upload metadata to the Hugging Face Hub.

    Args:
        metadata (dict): Metadata to upload.
        filename (str): Name of the file to upload.
        repo_id (str): Repository ID on the Hugging Face Hub.
        repo_dir (str): Directory in the repository to save the file.
    """
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
    """
    Push a folder to the Hugging Face Hub.

    Args:
        folder (str): Folder to push.
        repo_id (str): Repository ID on the Hugging Face Hub.
        branch (Optional[str]): Branch to push to.
    """
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


def convert_pdf2image(data_path: Path, output_dir: Path):
    """
    Convert PDF files to images using pdf2image.
    """
    return convert_from_path(data_path, output_folder=output_dir)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def base64_to_grayscale(base64_string: str) -> str:
    """
    Convert a base64 encoded image to grayscale and return as base64

    Args:
        base64_string (str): Base64 encoded image string
                            Can include data URL prefix or be raw base64

    Returns:
        str: Base64 encoded grayscale image
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Open image from bytes
        image = Image.open(BytesIO(image_bytes))

        # Convert to grayscale
        grayscale_image = image.convert("L")

        # Save grayscale image to bytes
        output_buffer = BytesIO()

        # Determine original format or default to PNG
        format = image.format if image.format else "PNG"
        grayscale_image.save(output_buffer, format=format)

        # Get bytes and encode to base64
        grayscale_bytes = output_buffer.getvalue()
        grayscale_base64 = base64.b64encode(grayscale_bytes).decode("utf-8")

        return grayscale_base64

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def prepare_image_and_text(
    local_pdf_path: str,
    page: int,
    target_dim: int = None,
    convert_to_grayscale: bool = True,
) -> tuple[str, str]:
    """
    Common utility to prepare image and anchor text from PDF.

    Returns:
        tuple: (image_base64, anchor_text)
    """
    # Use TARGET_IMAGE_DIM if target_dim not specified
    if target_dim is None:
        target_dim = TARGET_IMAGE_DIM

    image_base64 = render_pdf_to_base64png(local_pdf_path, page, target_dim)

    if convert_to_grayscale:
        image_base64 = base64_to_grayscale(image_base64)

    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    return image_base64, anchor_text


def load_prompt_template(prompt_key: str, prompt_path: str) -> Template:
    """
    Load and prepare prompt template from YAML file.

    Returns:
        Template: Jinja2 template ready for rendering
    """
    if not prompt_path or not os.path.exists(prompt_path):
        prompt_path = PROMPT_PATH

    with open(prompt_path, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)
        return Template(prompt_template_dict[prompt_key])


def create_vision_message(
    prompt_template: Template, anchor_text: Optional[str], image_base64: str
) -> list:
    """
    Create standardized message format for vision models.

    Returns:
        list: Message format compatible with OpenAI API
    """
    if anchor_text:
        rendered_prompt = prompt_template.render({"base_text": anchor_text})
    else:
        rendered_prompt = prompt_template.render()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": rendered_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ]


def print_results(prompt_template: Template, anchor_text: str, response_content: str):
    """
    Utility to print standardized results.
    """
    rendered_prompt = prompt_template.render({"base_text": anchor_text})
    print(f"Prompt: {rendered_prompt}\n" + "=" * 50)
    print(f"Response: {response_content}\n" + "=" * 50)

    try:
        parsed_response = json.loads(response_content)

        if isinstance(parsed_response, dict):
            if "natural_text" in parsed_response:
                print(f"Generated Natural Text: {parsed_response['natural_text']}")
        elif isinstance(parsed_response, list):
            for item in parsed_response:
                if isinstance(item, dict) and "natural_text" in item:
                    print(f"Generated Natural Text: {item['natural_text']}")
    except (json.JSONDecodeError, KeyError):
        print("Could not extract natural_text from response")


def openai_response_format_schema() -> dict:
    """
    Returns the OpenAI response format schema for page analysis tasks.
    This schema is used to validate the response format for tasks involving page analysis,
    such as determining the primary language, rotation validity, and content type of a page.
    """
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


def openai_response_format_schema_multipages() -> dict:
    """
    Returns the OpenAI response format schema for multiple page analysis tasks.
    This schema is used to validate the response format for tasks involving analysis of multiple pages,
    such as determining the primary language, rotation validity, and content type of each page.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pages_response",
            "schema": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "array",
                        "items": {
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
                        "description": "List of page analysis results",
                    },
                },
                "additionalProperties": False,
                "required": ["pages"],
            },
            "strict": True,
        },
    }


def text_order_response_format() -> dict:
    """
    Returns the OpenAI response format schema for text order test generation.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "text_order_response",
            "schema": {
                "type": "object",
                "properties": {
                    "tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_type": {
                                    "type": "string",
                                    "enum": ["text_order"],
                                    "description": "The type of test to be performed.",
                                },
                                "before": {
                                    "type": "string",
                                    "description": "The text that should appear before the target text.",
                                },
                                "after": {
                                    "type": "string",
                                    "description": "The text that should appear after the target text.",
                                },
                                "target": {
                                    "type": "string",
                                    "description": "The target text whose order is to be verified.",
                                },
                            },
                            "additionalProperties": False,
                            "required": ["test_type", "before", "after", "target"],
                        },
                        "description": "A list of tests to check for the presence of specific text in the image.",
                    }
                },
                "additionalProperties": False,
                "required": ["tests"],
            },
            "strict": True,
        },
    }


def text_present_response_format() -> dict:
    """
    Returns the OpenAI response format schema for text presence test generation.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "text_present_response",
            "schema": {
                "type": "object",
                "properties": {
                    "tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_type": {
                                    "type": "string",
                                    "enum": ["text_present"],
                                    "description": "The type of test to be performed.",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The text that is present in the image.",
                                },
                                "case_sensitive": {
                                    "type": "boolean",
                                    "description": "Indicates whether the text matching should be case sensitive.",
                                },
                                "first_n": {
                                    "type": ["string", "null"],
                                    "description": "If provided, only the first N characters of the text should be considered for matching. If null, consider the full text.",
                                },
                                "last_n": {
                                    "type": ["string", "null"],
                                    "description": "If provided, only the last N characters of the text should be considered for matching. If null, consider the full text.",
                                },
                            },
                            "additionalProperties": False,
                            "required": [
                                "test_type",
                                "text",
                                "case_sensitive",
                                "first_n",
                                "last_n",
                            ],
                        },
                        "description": "A list of tests to check for the presence of specific text in the image.",
                    }
                },
                "additionalProperties": False,
                "required": ["tests"],
            },
            "strict": True,
        },
    }


def text_absent_response_format() -> dict:
    """
    Returns the OpenAI response format schema for text absence test generation.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "text_absent_response",
            "schema": {
                "type": "object",
                "properties": {
                    "tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_type": {
                                    "type": "string",
                                    "enum": ["text_absent"],
                                    "description": "The type of test to be performed.",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The text that is present in the image.",
                                },
                                "case_sensitive": {
                                    "type": "boolean",
                                    "description": "Indicates whether the text matching should be case sensitive.",
                                },
                                "first_n": {
                                    "type": ["string", "null"],
                                    "description": "If provided, only the first N characters of the text should be considered for matching. If null, consider the full text.",
                                },
                                "last_n": {
                                    "type": ["string", "null"],
                                    "description": "If provided, only the last N characters of the text should be considered for matching. If null, consider the full text.",
                                },
                            },
                            "additionalProperties": False,
                            "required": [
                                "test_type",
                                "text",
                                "case_sensitive",
                                "first_n",
                                "last_n",
                            ],
                        },
                        "description": "A list of tests to check for the presence of specific text in the image.",
                    }
                },
                "additionalProperties": False,
                "required": ["tests"],
            },
            "strict": True,
        },
    }


class PageAnalysis(BaseModel):
    """
    Pydantic model representing the analysis results for a single page.
    """

    primary_language: Optional[str] = Field(
        None,
        description="The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
    )
    is_rotation_valid: bool = Field(
        description="Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures."
    )
    rotation_correction: Literal[0, 90, 180, 270] = Field(
        0,
        description="Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
    )
    is_table: bool = Field(
        description="Indicates if the majority of the page content is in tabular format."
    )
    is_diagram: bool = Field(
        description="Indicates if the majority of the page content is a visual diagram."
    )
    natural_text: Optional[str] = Field(
        None, description="The natural text content extracted from the page."
    )


class PagesAnalysisResponse(BaseModel):
    """
    Pydantic model representing the analysis results for multiple pages.
    """

    pages: List[PageAnalysis] = Field(description="List of page analysis results")
