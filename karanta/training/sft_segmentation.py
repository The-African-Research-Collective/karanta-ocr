#!/usr/bin/env python
# modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation.py

import logging
import os
import sys
from functools import partial

import albumentations as A
import evaluate
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from karanta.training.segmentation_args import (
    DataTrainingArguments,
    ModelArguments,
    ExperimentArguments,
)
from karanta.data.utils import prepare_mixed_datasets
from karanta.training.utils import ExtendedArgumentParser, reduce_labels_transform

logger = logging.getLogger(__name__)


def main(args: ExtendedArgumentParser):
    model_args, data_args, training_args = args[0], args[1], args[2]

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_semantic_segmentation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

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
            trust_remote_code=model_args.trust_remote_code,
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

    # Rename column names to standardized names (only "image" and "label" need to be present)
    if "pixel_values" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"pixel_values": "image"})
    if "annotation" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"annotation": "label"})

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = (
        None if "validation" in dataset.keys() else data_args.train_val_split
    )
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    id2label = {
        0: "whitespace",
        1: "articles",
    }
    label2id = {v: k for k, v in id2label.items()}

    # Load the mean IoU metric from the evaluate package
    metric = evaluate.load("mean_iou", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    @torch.no_grad()
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.do_reduce_labels,
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
        )

        return metrics

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        do_reduce_labels=data_args.do_reduce_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Define transforms to be applied to each image and target.
    if "shortest_edge" in image_processor.size:
        # We instead set the target size as (shortest_edge, shortest_edge) to here to ensure all images are batchable.
        height, width = (
            image_processor.size["shortest_edge"],
            image_processor.size["shortest_edge"],
        )
    else:
        height, width = image_processor.size["height"], image_processor.size["width"]
    train_transforms = A.Compose(
        [
            A.Lambda(
                name="reduce_labels",
                mask=reduce_labels_transform if data_args.do_reduce_labels else None,
                p=1.0,
            ),
            # Pad image with 255, because it is ignored by loss
            A.PadIfNeeded(
                min_height=height, min_width=width, border_mode=0, value=255, p=1.0
            ),
            A.Normalize(
                mean=image_processor.image_mean,
                std=image_processor.image_std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Lambda(
                name="reduce_labels",
                mask=reduce_labels_transform if data_args.do_reduce_labels else None,
                p=1.0,
            ),
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(
                mean=image_processor.image_mean,
                std=image_processor.image_std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )

    def preprocess_batch(example_batch, transforms: A.Compose):
        pixel_values = []
        labels = []
        for image, target in zip(example_batch["image"], example_batch["label"]):
            transformed = transforms(
                image=np.array(image.convert("RGB")), mask=np.array(target)
            )
            pixel_values.append(transformed["image"])
            labels.append(transformed["mask"])

        encoding = {}
        encoding["pixel_values"] = torch.stack(pixel_values).to(torch.float)
        encoding["labels"] = torch.stack(labels).to(torch.long)

        return encoding

    # Preprocess function for dataset should have only one argument,
    # so we use partial to pass the transforms
    preprocess_train_batch_fn = partial(preprocess_batch, transforms=train_transforms)
    preprocess_val_batch_fn = partial(preprocess_batch, transforms=val_transforms)

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(preprocess_train_batch_fn)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(preprocess_val_batch_fn)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=image_processor,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    parser = ExtendedArgumentParser(
        (ModelArguments, DataTrainingArguments, ExperimentArguments)
    )
    args = parser.parse()
    main(args)
