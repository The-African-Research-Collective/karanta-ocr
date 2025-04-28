from dataclasses import dataclass, field
from typing import Optional

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

    dataset_name: Optional[str] = field(
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
        print(self.dataset_name)
        if (
            self.dataset_name is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
            and self.train_dir is None
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
            or (self.dataset_name is not None and self.train_dir is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None)
                and self.train_dir is not None
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
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "Revision of the model to use (can be a branch name, tag name or git commit id)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to trust the remote code when loading the model."
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


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to the experiment configuration, replacing TrainingArguments.
    """

    output_dir: str = field(
        default="./output",
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
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hugging Face Hub."},
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set."},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
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

    def post__init__(self):
        if self.push_to_hub and self.hub_model_id is None:
            raise ValueError(
                "If `push_to_hub` is True, you must provide a `hub_model_id`."
            )
