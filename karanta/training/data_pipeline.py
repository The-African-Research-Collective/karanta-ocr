import logging

from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from karanta.training.utils import load_yaml_config

logger = logging.getLogger(__name__)


class Langages(Enum):
    """
    Enum for the languages supported in the pipeline
    languages are primarily African languages and some high resource languages
    """

    Yoruba = ["yo", "yor", "yoruba"]
    Igbo = ["ig", "ibo", "igbo"]
    Hausa = ["ha", "hau", "hausa"]
    Swahili = ["sw", "swa", "swahili"]
    Zulu = ["zu", "zul", "zulu"]
    English = ["en", "eng", "english"]
    French = ["fr", "fra", "french"]
    Arabic = ["ar", "ara", "arabic"]
    Spanish = ["es", "spa", "spanish"]
    Portuguese = ["pt", "por", "portuguese"]
    SouthernSotho = ["st", "sot", "sotho"]


@dataclass(frozen=True, slots=True)
class BasePipelineStep(ABC):
    """This is the base class for all the pipeline steps."""

    def __call__(self, *args, **kwargs):
        """Call the step with the given arguments."""
        pass


def initialize_dataset(
    json_dir: Path, pdf_dir: Path, max_workers: Optional[int] = None
) -> List[Tuple[Path, Path]]:
    json_files = list(json_dir.glob("*.json"))

    def check_pdf_wrapper(json_file: Path) -> Optional[Tuple[str, Tuple[Path, Path]]]:
        """
        Check if the corresponding PDF file exists for the given JSON file.
        """
        pdf_file = pdf_dir / f"{json_file.stem}.pdf"
        if pdf_file.exists():
            return json_file.stem, (pdf_file, json_file)
        return None

    pdf_files = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(check_pdf_wrapper, json_files)

        for result in results:
            if result is not None:
                name, (pdf_path, json_path) = result
                pdf_files[name] = (pdf_path, json_path)

    return list(pdf_files.values())


class LocalDataset(Dataset):
    """
    This is the dataset class for generating the dataset from a local directory for training and evaluation.
    This class locates the JSON files in a given directory and locates the corresponding PDF for that JSON file.
    Each pairs of JSON and PDF is treated as a single sample in the dataset and they go through a set of processing steps.

    These processing steps are defined in the config file and are executed in the order they are defined.
    """

    def __init__(
        self,
        root_dir: Path,
        pdf_dir_name: str = "pdf_inputs",
        json_dir_name: str = "json_outputs",
        pipeline_steps: Optional[List[BasePipelineStep]] = None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.pdf_dir = root_dir / pdf_dir_name
        self.json_dir = root_dir / json_dir_name

        self.dataset = initialize_dataset(self.json_dir, self.pdf_dir)

        self.pipeline_steps = pipeline_steps if pipeline_steps is not None else []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Fetch a single item from the dataset by index
        """
        sample = self.dataset[idx]

        for step in self.pipeline_steps:
            sample = step(sample)

        return sample


if __name__ == "__main__":
    config = load_yaml_config("configs/training/ocr/dummy.yaml")
    print(config)
    config = config["dataset"]["train"][0]
    dataset = LocalDataset(
        root_dir=Path(config["root_dir"]),
        pdf_dir_name=config["pdf_dir_name"],
        json_dir_name=config["json_dir_name"],
        pipeline_steps=None,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset samples: {dataset[0]}")
