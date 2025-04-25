import os
import sys
import dataclasses
from dataclasses import dataclass
import shutil

from typing import Optional, List, Any, Union, Tuple, NewType

from accelerate.logging import get_logger
from transformers import HfArgumentParser, TrainingArguments

from karanta.training.classification_args import ExperimentArguments

DataClassType = NewType("DataClassType", Any)
logger = get_logger(__name__)


class ExtendedArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_file: str, additional_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and override its values with command-line arguments.

        Args:
            yaml_file (str): Path to the YAML configuration file.
            additional_args (Optional[List[str]]): Additional command-line arguments.

        Returns:
            List[dataclass]: Parsed dataclasses with updated values.
        """
        yaml_args = self.parse_yaml_file(os.path.abspath(yaml_file))
        parsed_args = []

        additional_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1]
            for arg in additional_args or []
        }
        used_args = {}

        for yaml_data, data_class in zip(yaml_args, self.dataclass_types):
            keys = {field.name for field in dataclasses.fields(yaml_data) if field.init}
            inputs = {key: getattr(yaml_data, key) for key in keys}
            for arg, value in additional_args.items():
                if arg in keys:
                    base_type = yaml_data.__dataclass_fields__[arg].type
                    inputs[arg] = self._cast_type(value, base_type)
                    if arg in used_args:
                        raise ValueError(f"Duplicate argument provided: {arg}")
                    used_args[arg] = value
            parsed_args.append(data_class(**inputs))
            if len(parsed_args) == 1:
                parsed_args = parsed_args[0]
        return parsed_args

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            return self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            return self.parse_yaml_and_args(sys.argv[1], sys.argv[2:])
        return self.parse_args_into_dataclasses()

    @staticmethod
    def _cast_type(value: str, target_type: Any) -> Any:
        try:
            if target_type in [int, float]:
                return target_type(value)
            if target_type == List[str]:
                return value.split(",")
            if target_type is bool:
                return value.lower() in ["true", "1"]
            return value
        except ValueError as e:
            raise ValueError(
                f"Failed to cast value '{value}' to type {target_type}: {e}"
            )


def get_training_arguments(
    experiment_args: ExperimentArguments, training_args: TrainingArguments
) -> TrainingArguments:
    """
    Checks for attributes of ExperimentArguments in TrainingArguments and returns
    a TrainingArguments object with those non-null values from ExperimentArguments added.

    Args:
        experiment_args (ExperimentArguments): The ExperimentArguments object.
        training_args (TrainingArguments): The TrainingArguments object.

    Returns:
        TrainingArguments: A new TrainingArguments object with updated values.
    """
    updated_training_args = training_args.__class__(**vars(training_args))

    # Iterate over attributes in ExperimentArguments
    for attr, value in vars(experiment_args).items():
        # Check if the attribute exists in TrainingArguments and is not None
        if hasattr(updated_training_args, attr) and value is not None:
            setattr(updated_training_args, attr, value)

    return updated_training_args


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
