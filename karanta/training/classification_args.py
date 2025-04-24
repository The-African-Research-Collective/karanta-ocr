import logging
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_mixer: Optional[dict] = field(
        default=None,
        metadata={"help": "A dictionary of datasets (local or HF) to sample from."},
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None,
        metadata={"help": "A list of datasets (local or HF) to sample from."},
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the training data."}
    )
    validation_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the validation data."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={
            "help": "The name of the dataset column containing the image data. Defaults to 'image'."
        },
    )
    label_column_name: str = field(
        default="label",
        metadata={
            "help": "The name of the dataset column containing the labels. Defaults to 'label'."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
            and (self.train_dir is None and self.validation_dir is None)
        ):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )
        if (
            (
                self.dataset_name is not None
                and (
                    self.dataset_mixer is not None
                    or self.dataset_mixer_list is not None
                )
            )
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None)
                and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError(
                "You cannot specify a dataset name alongside a dataset mixer, dataset mixer list, or train/validation directory. Please choose only one data source configuration."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "Revision of the model to use (can be a branch name, tag name or git commit id)."
        },
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to the experiment configuration, replacing TrainingArguments.
    """

    output_dir: str = field(
        default="output",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite the content of the output directory. Use this to continue training a model."
        },
    )
    should_log: bool = field(
        default=True,
        metadata={"help": "Whether to log the training process."},
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Log every X updates steps."},
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "The logging level for the process. Options: 'info', 'debug', 'warning', 'error', 'critical'."
        },
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."
        },
    )
    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Run evaluation every X steps."},
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the best model found at the end of training."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "A seed for reproducible training."},
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision training."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hugging Face Hub."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank."},
    )
    disable_tqdm: bool = field(
        default=False,
        metadata={"help": "Whether to disable the tqdm progress bar."},
    )
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": "The integration to report the results and logs to. Options: 'tensorboard', 'wandb', etc."
        },
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set."},
    )
    hub_model_id: str = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    hub_token: str = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision training."},
    )

    def get_process_log_level(self) -> int:
        """
        Get the logging level for the process.

        Returns:
            int: The logging level (e.g., logging.INFO, logging.DEBUG).
        """
        return getattr(logging, self.log_level.upper(), logging.INFO)
