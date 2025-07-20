import os
import json
import logging
import time

from typing import List, Optional, Union
from pathlib import Path
from functools import wraps


from pdf2image import convert_from_path
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_fixed


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
