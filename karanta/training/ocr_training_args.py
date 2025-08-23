from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any


@dataclass
class ExperimentArguments:
    """
    Training arguments for training the vision language model (VLM) for OCR
    """

    exp_name: str = field(metadata={"help": "The name of the experiment"})
    logging_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Log every n steps."},
    )
    output_dir: str = field(
        default="output/",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    eval_strategy: str = field(
        default="steps",
        metadata={
            "help": "The evaluation strategy to use. Possible values are 'no', 'epoch', and 'steps'."
        },
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Run an evaluation every n steps."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    optimizer: str = field(
        default="adamw_torch",
        metadata={
            "help": "The optimizer to use. Possible values are 'adamw_torch', 'adamw_torch_fused', and 'adamw_apex_fused'."
        },
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether to push the model to the hub"}
    )
    hf_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "The huggingface repository id to push the model to"},
    )
    hf_entity: Optional[str] = field(
        default=None, metadata={"help": "The huggingface entity to push the model to"}
    )
    hf_repo_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The huggingface repository revision to push the model to"},
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "The seed to use for the run"}
    )
    reports_to: Optional[List[str]] = field(
        default="wandb", metadata={"help": "The service to report to"}
    )
    timeout: Optional[int] = field(
        default=600, metadata={"help": "The timeout for the run"}
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of gradient accumulation steps to use."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name to use for logging to wandb."},
    )
    use_flash_attention: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use flash attention"}
    )
    with_tracking: bool = field(
        default=True,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The version of the model on huggingface to use"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The model configuration to use, it is usually a model name"},
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Whether to trust remote code"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer to use"}
    )
    tokenizer_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The version of the tokenizer on huggingface to use"},
    )
    use_slow_tokenizer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a slow tokenizer"}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use the LORA training"}
    )
    lora_rank: Optional[int] = field(
        default=64, metadata={"help": "The rank of the LORA model"}
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    use_qlora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use the qlora training"}
    )
    add_bos_token: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add a beginning of sentence token"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing"}
    )


@dataclass
class DatasetArguments:
    dataset_eval: List[Dict[str, Any]] = field(default_factory=list)
    dataset_train: List[Dict[str, Any]] = field(default_factory=list)
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "The number of workers to use for data loading."},
    )
    max_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum sequence length for the model. "
            "Sequences will be truncated to this length."
        },
    )
