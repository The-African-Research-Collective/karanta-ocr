from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

from karanta.training.utils import load_yaml_config, SingleDatapoint
from karanta.training.pipeline_steps import BasePipelineStep, PDF2ImageStep

str2PipelineStep = {
    "PDF2ImageStep": PDF2ImageStep,
}


def initialize_dataset(
    json_dir: Path, pdf_dir: Path, max_workers: Optional[int] = None
) -> List[SingleDatapoint]:
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

    return [
        SingleDatapoint(pdf_path=pdf_path, json_path=json_path)
        for pdf_path, json_path in pdf_files.values()
    ]


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
        pipeline_steps: Optional[List[Dict]] = None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.pdf_dir = root_dir / pdf_dir_name
        self.json_dir = root_dir / json_dir_name

        self.dataset: List[SingleDatapoint] = initialize_dataset(
            self.json_dir, self.pdf_dir
        )
        self.pipeline = self._initialize_pipeline_steps(pipeline_steps)

    def _initialize_pipeline_steps(
        self, pipeline_config: List[Dict]
    ) -> List[BasePipelineStep]:
        """
        Initialize the pipeline steps from the configuration list.
        """
        pipeline = []
        for step_config in pipeline_config:
            name = step_config.pop("name")
            step_class = str2PipelineStep.get(name)
            step_instance = step_class(**step_config)
            pipeline.append(step_instance)

        return pipeline

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> SingleDatapoint:
        """
        Fetch a single item from the dataset by index
        """
        sample = self.dataset[idx]

        for step in self.pipeline:
            sample = step(sample)

        return sample


if __name__ == "__main__":
    config = load_yaml_config("configs/training/ocr/dummy.yaml")
    # print(config)
    config = config["dataset"]["train"][0]
    pipeline = config["pipeline"]
    dataset = LocalDataset(
        root_dir=Path(config["root_dir"]),
        pdf_dir_name=config["pdf_dir_name"],
        json_dir_name=config["json_dir_name"],
        pipeline_steps=pipeline,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset samples: {dataset[0]}")
