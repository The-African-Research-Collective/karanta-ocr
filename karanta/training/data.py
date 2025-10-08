import json
import torch
import numpy as np

from pathlib import Path
from transformers import AutoProcessor
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence

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
        num_samples: Optional[int] =  -1
    ):
        super().__init__()

        self.root_dir = root_dir
        self.pdf_dir = root_dir / pdf_dir_name
        self.json_dir = root_dir / json_dir_name

        self.dataset: List[SingleDatapoint] = initialize_dataset(
            self.json_dir, self.pdf_dir
        )
        if isinstance(num_samples, int) and num_samples != -1:
            self.dataset = self.dataset[:num_samples]

        self.pipeline = self._initialize_pipeline_steps(pipeline_steps)

        self.cache_dir = Path(self.root_dir) / ".cache" / "processed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

        cache_path =  self.cache_dir / f"{sample.pdf_path.stem}.pkl"
        if cache_path.exists():
            return torch.load(cache_path)
        else:
            for step in self.pipeline:
                sample = step(sample)

            model_inputs = sample.model_inputs
            torch.save(model_inputs, cache_path)
            return model_inputs


class DataCollator:
    """
    Prepares a batch of multimodal OCR samples for training Qwen2.5-VL-3B-Instruct.
    Handles tokenizer-based padding for text and dynamic padding for image embeddings.
    """

    def __init__(self, model_name_or_path: str, max_token_len: Optional[int] = None):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.max_token_len = max_token_len
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.masking_index = -100  # For label ignore

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, attention_masks_list, labels_list = [], [], []
        pixel_values_list, image_grid_thw_list = [], []

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x

        for sample in batch:
            if not sample:
                continue

            input_ids = to_tensor(sample["input_ids"]).squeeze(0)
            attention_mask = to_tensor(sample["attention_mask"]).squeeze(0)
            labels = to_tensor(sample["labels"]).squeeze(0)

            if self.max_token_len is not None:
                input_ids = input_ids[: self.max_token_len]
                attention_mask = attention_mask[: self.max_token_len]
                labels = labels[: self.max_token_len]

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            labels_list.append(labels)

            # pixel_values may be [num_patches, hidden_dim] → variable num_patches
            pixel_values = to_tensor(sample["pixel_values"])
            pixel_values_list.append(pixel_values)

            if "image_grid_thw" in sample:
                image_grid_thw_list.append(to_tensor(sample["image_grid_thw"]))

        if not input_ids_list:
            return {}

        # === Pad text using tokenizer ===
        batch_encodings = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_masks_list,
            },
            padding=True,
            return_tensors="pt",
        )

        labels_padded = self.processor.tokenizer.pad(
            {"input_ids": labels_list},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels_padded[labels_padded == self.pad_token_id] = self.masking_index

        # === Pad visual embeddings ===
        # pixel_values: list of [num_patches, hidden_dim] tensors → pad along num_patches
        pixel_values_padded = pad_sequence(pixel_values_list, batch_first=True, padding_value=0.0)

        image_grid_thw = torch.stack(image_grid_thw_list) if image_grid_thw_list else None

        batch_dict = {
            "input_ids": batch_encodings["input_ids"],
            "attention_mask": batch_encodings["attention_mask"],
            "labels": labels_padded,
            "pixel_values": pixel_values_padded,
        }

        if image_grid_thw is not None:
            batch_dict["image_grid_thw"] = image_grid_thw

        return batch_dict


if __name__ == "__main__":
    all_config = load_yaml_config("configs/training/ocr/karanta_set_qwen_2_5_3B_vl.yaml")
    # print(all_config)
    config = all_config["dataset_train"][3]
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
    # for i in range(len(dataset)):
    #     print(dataset[i]['input_ids'])
    #     print(dataset[i]['input_ids'].shape)
    # print(processor.tokenizer.pad_token_id)
    # print(processor.tokenizer.padding_side)

    # print(processor.tokenizer.decode(dataset[0]['input_ids']))

    # print(processor.tokenizer.pad_token_id in dataset[0]['input_ids'])

    # print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=DataCollator("Qwen/Qwen2.5-VL-3B-Instruct", max_token_len=all_config["max_length"]),
        num_workers=16,
    )
    from tqdm import tqdm

    for b in tqdm(dataloader):
        continue

    # print(next(iter(dataloader))['input_ids'].shape)

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
