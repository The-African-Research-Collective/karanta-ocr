import json
import torch
import numpy as np

from pathlib import Path
from transformers import AutoProcessor
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

from karanta.training.utils import load_yaml_config, SingleDatapoint
from karanta.training.pipeline_steps import (
    BasePipelineStep,
    PDF2ImageStep,
    JSONOutputFormat,
    FetchPageData,
    StaticLengthDocumentAnchoring,
    FinetuningPrompt,
    InstructUserMessages,
    Tokenizer,
    FetchMultipageData
)

str2PipelineStep = {
    "PDF2ImageStep": PDF2ImageStep,
    "JSONOutputFormat": JSONOutputFormat,
    "FetchPageData": FetchPageData,
    "StaticLengthDocumentAnchoring": StaticLengthDocumentAnchoring,
    "FinetuningPrompt": FinetuningPrompt,
    "InstructUserMessages": InstructUserMessages,
    "Tokenizer": Tokenizer,
    "FetchMultipageData": FetchMultipageData
}

def check_tokens_and_labels(input_ids, labels):
    # Count total tokens in input_ids
    total_tokens = input_ids.numel()
    
    # Find all non -100 items in labels
    non_padding_mask = labels != -100
    non_padding_items = labels[non_padding_mask]
    valid_label_count = non_padding_items.numel()
    
    # Print results
    print(f"Total tokens in input_ids: {total_tokens}")
    print(f"Valid labels (non -100): {valid_label_count}")
    print(f"Percentage of valid labels: {(valid_label_count / total_tokens * 100):.2f}%")
    
    return total_tokens, valid_label_count, non_padding_items


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
            # Check that json text loads successfully
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    _ = json.loads(f.read())["generation"]["pages"]
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"Error reading JSON file: {json_file}")

                with open("failed_files.txt", "a") as failed:
                    failed.write(str(json_file) + "\n")

                return None

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

            if name == "Tokenizer":
                processor = step_config.pop("processor", None)
                processor = AutoProcessor.from_pretrained(
                    processor
                )
                processor.num_additional_image_tokens = 1
                processor.num_additional_tokens = 1

                step_class = str2PipelineStep.get(name)
                step_instance = step_class(**step_config, processor=processor)
            else:
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

        return sample.model_inputs


class DataCollator:
    """
    The data collator is used to prepare a batch of data for training.
    This data collator accepts a list of Dicts, each representing a sample,
    and processes them according to the pipeline steps defined in the dataset.
    """

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        image_grid_thw = []

        for sample in batch:
            if sample:
                sample_input_ids = sample["input_ids"]
                sample_attention_mask = sample["attention_mask"]
                sample_labels = sample["labels"]

                if isinstance(sample_input_ids, np.ndarray):
                    input_ids_tensor = torch.from_numpy(sample_input_ids)
                elif isinstance(sample_input_ids, torch.Tensor):
                    input_ids_tensor = sample_input_ids
                else:
                    input_ids_tensor = torch.tensor(sample_input_ids)

                input_ids.append(input_ids_tensor[: self.max_token_len])

                if isinstance(sample_attention_mask, np.ndarray):
                    attention_mask_tensor = torch.from_numpy(sample_attention_mask)
                elif isinstance(sample_attention_mask, torch.Tensor):
                    attention_mask_tensor = sample_attention_mask
                else:
                    attention_mask_tensor = torch.tensor(sample_attention_mask)

                attention_mask.append(attention_mask_tensor[: self.max_token_len])

                if isinstance(sample_labels, np.ndarray):
                    labels_tensor = torch.from_numpy(sample_labels)
                elif isinstance(sample_labels, torch.Tensor):
                    labels_tensor = sample_labels
                else:
                    labels_tensor = torch.tensor(sample_labels)

                labels.append(labels_tensor[: self.max_token_len])

                # Handle pixel_values which might be numpy array or already a tensor
                sample_pixel_values = sample["pixel_values"]
                if isinstance(sample_pixel_values, np.ndarray):
                    sample_pixel_values = torch.from_numpy(sample_pixel_values)
                pixel_values.append(sample_pixel_values)

                # Handle image_grid_thw
                sample_image_grid_thw = sample["image_grid_thw"]
                if isinstance(sample_image_grid_thw, np.ndarray):
                    sample_image_grid_thw = torch.from_numpy(sample_image_grid_thw)
                image_grid_thw.append(sample_image_grid_thw)

        if not input_ids:
            return

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack(pixel_values),
            "image_grid_thw": torch.stack(image_grid_thw),
        }


if __name__ == "__main__":
    all_config = load_yaml_config("configs/training/ocr/karanta_set_qwen_2_5_3B_vl.yaml")
    print(all_config)
    config = all_config["dataset_train"][0]
    pipeline = config["pipeline"]
    dataset = LocalDataset(
        root_dir=Path(config["root_dir"]),
        pdf_dir_name=config["pdf_dir_name"],
        json_dir_name=config["json_dir_name"],
        pipeline_steps=pipeline,
    )

    from transformers import AutoProcessor

    # torch.set_printoptions(threshold=10_000)
    # numpy.set_printoptions(threshold=10_000)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print(processor.tokenizer.padding_side)

    # print(
    #     processor(text = ["I am the prince of wales"], return_tensors="np",padding="max_length", max_length=200, )
    # )

    # print(dataset[0]['input_ids'])
    # print(dataset[0]['input_ids'].shape)
    # print(processor.tokenizer.pad_token_id)
    # print(processor.tokenizer.padding_side)

    # print(processor.tokenizer.pad_token_id in dataset[0]['input_ids'])

    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=DataCollator(max_token_len=all_config["max_length"]),
        num_workers=all_config["dataloader_num_workers"],
    )

    next(iter(dataloader))

    # from tqdm import tqdm

    # for sample in tqdm(iter(dataloader), total=len(dataset)):

    #     print(sample['input_ids'].shape)
    #     print(sample['attention_mask'].shape)
    #     print(sample['labels'].shape)
    #     print(torch.sum(sample['attention_mask']))

    #     check_tokens_and_labels(sample['input_ids'], sample['labels'])
    #     print("====================================")

    # print(f"Dataset samples: {dataset[0].user_messages}")

    # from transformers import AutoProcessor

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # text = processor.apply_chat_template(
    #     [dataset[0].user_messages],  # Wrap in list as expected by the template
    #     tokenize=False,  # Keep as text, don't tokenize yet
    #     add_generation_prompt=True,  # Add generation tokens depending on the model
    # )
    # print(text)
