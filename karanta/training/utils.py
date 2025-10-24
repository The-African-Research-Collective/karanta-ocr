import os
import sys
import torch
import logging
import shutil
import dataclasses

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from PIL import Image
from collections import OrderedDict

from typing import Optional, List, Any, Union, Tuple, NewType

from accelerate import Accelerator
from transformers import HfArgumentParser, TrainingArguments

from karanta.training.classification_args import ExperimentArguments

DataClassType = NewType("DataClassType", Any)
logger = logging.getLogger(__name__)


class Languages(Enum):
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
    response: Optional[str] = None
    page_data: Optional[dict] = None
    anchor_text: Optional[str] = None
    instruction_prompt: Optional[str] = None
    user_messages: Optional[str] = None
    model_inputs: dict = field(default_factory=dict)


class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # noqa adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
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
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


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


def get_last_checkpoint_path(args, incomplete: bool = False) -> str:
    # if output already exists and user does not allow overwriting, resume from there.
    # otherwise, resume if the user specifies a checkpoint.
    # else, start from scratch.
    # if incomplete is true, include folders without "COMPLETE" in the folder.
    last_checkpoint_path = None
    if (
        args.output_dir
        and os.path.isdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        last_checkpoint_path = get_last_checkpoint(
            args.output_dir, incomplete=incomplete
        )
        if last_checkpoint_path is None:
            logger.warning(
                "Output directory exists but no checkpoint found. Starting from scratch."
            )
    elif args.resume_from_checkpoint:
        last_checkpoint_path = args.resume_from_checkpoint

    return last_checkpoint_path


def save_with_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: str,
    use_lora: bool = False,
    model_attribute_to_save: Optional[str] = None,
) -> None:
    """`model_attribute_to_save` is used to save PPO's policy instead of the full model"""
    # set the generation config to an empty setting to be safe.
    # we usually do greedy decoding for generation, so this should be okay.
    # otherwise, we get an error thrown at save time.
    unwrapped_model = accelerator.unwrap_model(model)
    if model_attribute_to_save is not None:
        unwrapped_model = getattr(unwrapped_model, model_attribute_to_save)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)

    # if we are saving a specific attribute of the model, we need to filter the state_dict
    # also the state_dict only lives in the main process; other processes just have state_dict = None
    if model_attribute_to_save is not None and accelerator.is_main_process:
        state_dict = OrderedDict(
            {
                k[len(f"{model_attribute_to_save}.") :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_attribute_to_save}.")
            }
        )

    if use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )


def is_checkpoint_folder(dir: str, folder: str) -> bool:
    return (
        folder.startswith("step_") or folder.startswith("epoch_")
    ) and os.path.isdir(os.path.join(dir, folder))


def clean_last_n_checkpoints(output_dir: str, keep_last_n_checkpoints: int) -> None:
    # remove the last checkpoint to save space
    folders = [f for f in os.listdir(output_dir) if is_checkpoint_folder(output_dir, f)]
    # find the checkpoint with the largest step
    checkpoints = sorted(folders, key=lambda x: int(x.split("_")[-1]))
    if keep_last_n_checkpoints != -1 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[: len(checkpoints) - keep_last_n_checkpoints]:
            logger.info(f"Removing checkpoint {checkpoint}")
            shutil.rmtree(os.path.join(output_dir, checkpoint))
    logger.info("Remaining files:" + str(os.listdir(output_dir)))
