#!/usr/bin/env python
# modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py

"""Finetuning 🤗 Transformers model for instance segmentation leveraging the Trainer API."""

import logging
import os
import sys
import pathlib
from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Tuple, Union

import numpy as np
import albumentations as A
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from PIL import Image
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    Trainer,
    TrainingArguments,
)

from transformers.image_processing_utils import BatchFeature
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.image_utils import get_image_size, ChannelDimension

from karanta.training.segmentation_args import (
    DataTrainingArguments,
    ModelArguments,
    ExperimentArguments,
    ModelOutput,
)
from karanta.training.utils import ExtendedArgumentParser

logger = logging.getLogger(__name__)


# Modified from transformers.models.vilt.image_processing_vilt.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray,
    output_size: Tuple[int, int],
    masks_path: pathlib.Path,
    file_name: str,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
        masks_path (`pathlib.Path`):
            Directory to save the mask.
        file_name (`str`):
            Name of the file to save the mask.
    """
    save_path = pathlib.Path(masks_path) / file_name
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    Image.fromarray(mask.astype(np.uint8)).save(save_path)


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,  # Updated processor
    masks_path: pathlib.Path,
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for idx, (pil_image, annotation_dict) in enumerate(
        zip(examples["image"], examples["label"])
    ):
        image = np.array(pil_image)
        output_size = get_image_size(image)
        make_pixel_mask(
            image=image,
            output_size=output_size,
            masks_path=masks_path,
            file_name=annotation_dict["file_name"],
        )
        # generate masks
        target_with_masks = image_processor.prepare_annotation(
            image=image, target=annotation_dict, masks_path=masks_path
        )
        output = transform(
            image=image, annotation_with_masks=target_with_masks["masks"]
        )

        # Preprocess with the processor using the annotation as-is
        model_inputs = image_processor(
            images=output["image"],
            annotations=output["annotation_with_masks"],
            return_tensors="pt",
        )
        print(model_inputs.keys())
        batch["pixel_values"].append(model_inputs["pixel_values"][0])
        batch["mask_labels"].append(model_inputs["mask_labels"][0])
        batch["class_labels"].append(model_inputs["class_labels"][0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack(
        [example["pixel_values"] for example in examples]
    )
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack(
            [example["pixel_mask"] for example in examples]
        )
    return batch


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


class Evaluator:
    """
    Compute metrics for the instance segmentation task.
    """

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.0,
    ):
        """
        Initialize evaluator with image processor, id2label mapping and threshold for filtering predictions.

        Args:
            image_processor (AutoImageProcessor): Image processor for
                `post_process_instance_segmentation` method.
            id2label (Mapping[int, str]): Mapping from class id to class name.
            threshold (float): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        return metric

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> list[dict[str, torch.Tensor]]:
        """Collect targets in a form of list of dictionaries with keys "masks", "labels"."""
        batch_masks = target_batch[0]
        batch_labels = target_batch[1]
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append(
                {
                    "masks": masks.to(dtype=torch.bool),
                    "labels": labels,
                }
            )
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> list[list[int]]:
        target_sizes = []
        for target in post_processed_targets:
            target_sizes.append(target["masks"].shape[-2:])
        return target_sizes

    def postprocess_prediction_batch(
        self, prediction_batch, target_sizes
    ) -> list[dict[str, torch.Tensor]]:
        """Collect predictions in a form of list of dictionaries with keys "masks", "labels", "scores"."""

        model_output = ModelOutput(
            class_queries_logits=prediction_batch[0],
            masks_queries_logits=prediction_batch[1],
        )
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor(
                        [x["label_id"] for x in image_predictions["segments_info"]]
                    ),
                    "scores": torch.tensor(
                        [x["score"] for x in image_predictions["segments_info"]]
                    ),
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(
        self, evaluation_results: EvalPrediction, compute_result: bool = False
    ) -> Mapping[str, float]:
        """
        Update metrics with current evaluation results and return metrics if `compute_result` is True.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            compute_result (bool): Whether to compute and return metrics.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"
        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(
            prediction_batch, target_sizes
        )

        # Compute metrics
        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                self.id2label[class_id.item()]
                if self.id2label is not None
                else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        # Reset metric for next evaluation
        self.reset_metric()

        return metrics


def setup_logging(training_args: TrainingArguments) -> None:
    """Setup logging according to `training_args`."""

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


def find_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    """Find the last checkpoint in the output directory according to parameters specified in `training_args`."""

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return checkpoint


def main(args: ExtendedArgumentParser):
    model_args, data_args, training_args = args[0], args[1], args[2]

    if isinstance(training_args.learning_rate, str):
        try:
            training_args.learning_rate = float(training_args.learning_rate)
        except ValueError:
            raise ValueError(
                f"Learning rate: {training_args.learning_rate} is not a float. Please provide a float"
            )
    training_args.eval_do_concat_batches = False
    training_args.batch_eval_metrics = True
    training_args.remove_unused_columns = False

    masks_dir = pathlib.Path(training_args.output_dir) / "masks"
    os.makedirs(masks_dir, exist_ok=True)

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_instance_segmentation", model_args, data_args)

    # Setup logging and log on each process the small summary:
    setup_logging(training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load last checkpoint from output_dir if it exists (and we are not overwriting it)
    checkpoint = find_last_checkpoint(training_args)

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(
        data_args.dataset_name, trust_remote_code=model_args.trust_remote_code
    )

    if "pixel_values" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"pixel_values": "image"})
    if "annotations" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"annotations": "label"})

    if "validation" not in dataset:
        dataset["validation"] = dataset["train"].train_test_split(test_size=0.15)

    # We need to specify the label2id mapping for the model
    # it is a mapping from semantic class name to class index.
    # In case your dataset does not provide it, you can create it manually:
    label2id = {"whitespace": 0, "article": 1}

    if data_args.do_reduce_labels:
        label2id = {
            name: idx for name, idx in label2id.items() if idx != 0
        }  # remove background class
        label2id = {
            name: idx - 1 for name, idx in label2id.items()
        }  # shift class indices by -1

    id2label = {v: k for k, v in label2id.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------
    model = AutoModelForUniversalSegmentation.from_pretrained(
        model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=model_args.token,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        do_resize=True,
        size={"height": model_args.image_height, "width": model_args.image_width},
        format="coco_panoptic",
        do_reduce_labels=data_args.do_reduce_labels,
        reduce_labels=data_args.do_reduce_labels,
        token=model_args.token,
    )

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    train_augment_and_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        is_check_shapes=False,
    )
    validation_transform = A.Compose(
        [A.NoOp()],
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=image_processor,
        masks_path=masks_dir,
    )
    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor,
    )
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(
        validation_transform_batch
    )

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    compute_metrics = Evaluator(
        image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(
            eval_dataset=dataset["validation"], metric_key_prefix="test"
        )
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-segmentation", "instance-segmentation", "vision"],
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
