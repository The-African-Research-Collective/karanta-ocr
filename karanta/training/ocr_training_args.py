from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class TrainingArguments:
    """
    Training arguments for training the vision language model (VLM) for OCR
    """

    run_name: str = field(
        default=None, metadata={"help": "The name of the experiment run"}
    )
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
