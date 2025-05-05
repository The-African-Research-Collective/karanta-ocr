import os
import sys
import logging
import dataclasses
from dataclasses import dataclass
import shutil
from typing import Optional, List, Any, Union, Tuple, NewType

import cv2
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import HfArgumentParser, TrainingArguments
from scipy.ndimage import label, find_objects

from karanta.training.classification_args import ExperimentArguments

DataClassType = NewType("DataClassType", Any)
logger = logging.getLogger(__name__)


class ExtendedArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used.
            other_args (`List[str]`, *optional*):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: A list of dataclasses with the values from the YAML file and the command line.
        """
        # Parse the YAML file
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args or []
        }
        used_args = {}

        # Overwrite the default/loaded values with the values provided via the command line
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_class) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                if arg in keys:
                    base_type = data_class.__dataclass_fields__[arg].type
                    # Cast type for ints, floats, lists, and bools
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)
                    elif base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]
                    elif base_type is bool:
                        inputs[arg] = val.lower() in ["true", "1"]
                    else:
                        inputs[arg] = val

                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        """
        Parse arguments from a YAML configuration file or command-line arguments.

        Returns:
            Union[DataClassType, Tuple[DataClassType]]: Parsed dataclasses.
        """
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        else:
            # Parse command-line arguments only
            output = self.parse_args_into_dataclasses()

        if len(output) == 3:
            model_args, data_args, experiment_args = output
            training_args = TrainingArguments(output_dir=experiment_args.output_dir)
            training_args = get_training_arguments(experiment_args, training_args)
            return model_args, data_args, training_args

        # Return a single dataclass if only one is parsed
        if len(output) == 1:
            output = output[0]
        return output


def get_training_arguments(
    experiment_args: ExperimentArguments, training_args: TrainingArguments
) -> TrainingArguments:
    """
    Updates a TrainingArguments object with non-null values from ExperimentArguments.

    Args:
        experiment_args (ExperimentArguments): The ExperimentArguments object.
        training_args (TrainingArguments): The TrainingArguments object.

    Returns:
        TrainingArguments: The updated TrainingArguments object.
    """
    # Update attributes with non-null values from ExperimentArguments
    for attr, value in vars(experiment_args).items():
        if hasattr(training_args, attr) and value is not None:
            setattr(training_args, attr, value)

    return training_args


def get_last_checkpoint(folder: str, incomplete: bool = False) -> Optional[str]:
    """
    Retrieve the last checkpoint from a folder.

    Args:
        folder (str): Path to the folder containing checkpoints.
        incomplete (bool): Include incomplete checkpoints if True.

    Returns:
        Optional[str]: Path to the last checkpoint.
    """
    checkpoints = [f for f in os.listdir(folder) if f.startswith(("step_", "epoch_"))]
    if not incomplete:
        checkpoints = [
            ckpt
            for ckpt in checkpoints
            if os.path.exists(os.path.join(folder, ckpt, "COMPLETED"))
        ]
    return (
        os.path.join(folder, max(checkpoints, key=lambda x: int(x.split("_")[-1])))
        if checkpoints
        else None
    )


def clean_old_checkpoints(output_dir: str, keep_last_n: int) -> None:
    """
    Remove old checkpoints to save space.

    Args:
        output_dir (str): Directory containing checkpoints.
        keep_last_n (int): Number of recent checkpoints to keep.
    """
    checkpoints = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(("step_", "epoch_"))],
        key=lambda x: int(x.split("_")[-1]),
    )
    for checkpoint in checkpoints[:-keep_last_n]:
        shutil.rmtree(os.path.join(output_dir, checkpoint))
        logger.info(f"Removed checkpoint: {checkpoint}")


def reduce_labels_transform(labels: np.ndarray, **kwargs) -> np.ndarray:
    """Set `0` label (whitespace) to 255 (ignored) and keep other labels unchanged.

    Example:
        Initial class labels:         0 - whitespace; 1 - tables; 2 - logos; 3 - photos; 4 - articles;
        Transformed class labels:   255 - whitespace; 1 - tables; 2 - logos; 3 - photos; 4 - articles;

    **kwargs are required to use this function with albumentations.
    """
    labels[labels == 0] = 255  # Set the "whitespace" label to 255 (ignored by loss)
    return labels


def masks_to_bounding_boxes(segmentation_mask):
    """
    Converts a segmentation mask into bounding boxes.

    Args:
        segmentation_mask (numpy.ndarray): The segmentation mask output by the model.
                                           Each pixel has a class label (e.g., 0 for background, 1 for articles).

    Returns:
        bounding_boxes (list): A list of bounding boxes, where each box is represented as
                               (label, x_min, y_min, x_max, y_max).
    """
    bounding_boxes = []

    # Exclude the background (label 0)
    for label_id in np.unique(segmentation_mask):
        if label_id == 0:  # Skip background
            continue

        # Create a binary mask for the current label
        binary_mask = segmentation_mask == label_id

        # Find connected components
        labeled_mask, num_features = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        for i, slice_ in enumerate(slices):
            if slice_ is not None:
                y_min, y_max = slice_[0].start, slice_[0].stop
                x_min, x_max = slice_[1].start, slice_[1].stop
                bounding_boxes.append((label_id, x_min, y_min, x_max, y_max))

    return bounding_boxes


def postprocess_and_get_bounding_boxes(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # Scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()

    # Convert masks to bounding boxes
    all_bounding_boxes = []
    for mask in pred_labels:
        bounding_boxes = masks_to_bounding_boxes(mask)
        all_bounding_boxes.append(bounding_boxes)

    return all_bounding_boxes


def visualize_bounding_boxes(image, bounding_boxes):
    """
    Visualizes bounding boxes on the original image.

    Args:
        image (numpy.ndarray): The original image.
        bounding_boxes (list): A list of bounding boxes (label, x_min, y_min, x_max, y_max).
    """
    image_with_boxes = image.copy()
    for label_id, x_min, y_min, x_max, y_max in bounding_boxes:
        # Draw the bounding box
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # Add the label
        cv2.putText(
            image_with_boxes,
            f"Label {label_id}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Display the image
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.show()
