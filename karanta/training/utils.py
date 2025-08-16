import os
import sys
import logging
import shutil
import dataclasses

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image

from typing import Optional, List, Any, Union, Tuple, NewType

from transformers import HfArgumentParser, TrainingArguments

from karanta.training.classification_args import ExperimentArguments

DataClassType = NewType("DataClassType", Any)
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


@dataclass(slots=True)
class SingleDatapoint:
    pdf_path: Path
    json_path: Path
    image: Optional[Image.Image] = None


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


def load_yaml_config(yaml_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML configuration.
    """
    import yaml

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config
