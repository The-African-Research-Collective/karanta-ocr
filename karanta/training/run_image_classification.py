#!/usr/bin/env python
# modified from https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification
import logging
import os
import sys

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Resize,
    ToTensor,
)

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from karanta.training.classification_args import (
    DataTrainingArguments,
    ModelArguments,
    ExperimentArguments,
)
from karanta.data.utils import prepare_mixed_datasets
from karanta.training.utils import ExtendedArgumentParser

logger = logging.getLogger(__name__)


def main(args: ExtendedArgumentParser):
    model_args, data_args, training_args = args[0], args[1], args[2]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_mixer or data_args.dataset_mixer_list:
        dataset = prepare_mixed_datasets(
            dataset_sources=data_args.dataset_mixer or data_args.dataset_mixer_list,
            splits=["train", "validation"],
            save_dir=data_args.output_dir,
            shuffle_data=True,
            ensure_columns=[data_args.image_column_name, data_args.label_column_name],
            include_source_column=True,
        )
    elif data_args.dataset_name:
        dataset = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_dir:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = (
        dataset["train"].column_names
        if "train" in dataset
        else dataset["validation"].column_names
    )
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct column."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct column."
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor(
            [example[data_args.label_column_name] for example in examples]
        )
        return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    if "validation" not in dataset.keys() and data_args.train_val_split:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )
    else:
        normalize = Lambda(lambda x: x)

    _train_transforms = Compose(
        [
            Resize(image_processor.size["shortest_edge"]),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(image_processor.size["shortest_edge"]),
            CenterCrop(image_processor.size["shortest_edge"]),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    if training_args.do_train:
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        dataset["validation"].set_transform(val_transforms)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    parser = ExtendedArgumentParser(
        (ModelArguments, DataTrainingArguments, ExperimentArguments)
    )
    args = parser.parse()
    main(args)
