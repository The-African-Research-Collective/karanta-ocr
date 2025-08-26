import os
import time
import random
import logging
import torch
import math
import transformers

from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict
from torch.utils.data import ConcatDataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration
from transformers import (
    AutoConfig,
    AutoModel,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from karanta.training.utils import ArgumentParserPlus, get_last_checkpoint_path
from karanta.training.ocr_training_args import (
    ExperimentArguments,
    ModelArguments,
    DatasetArguments,
)
from karanta.training.data import LocalDataset, DataCollator

load_dotenv()
logger = get_logger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloader: Dict[str, DataLoader],
    device: torch.device,
    reduce_loss: str,
    device_type: str,
    dtype: str,
):
    model.eval()

    with torch.no_grad():
        for eval_set, eval_set_dataloader in eval_dataloader.items():
            total_loss = 0.0
            num_batches = 0

            for batch in eval_set_dataloader:
                if batch is None:
                    continue

                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device_type, dtype=dtype):
                    outputs = model(**batch)

    return total_loss, num_batches, outputs


def main(args: ArgumentParserPlus):
    exp_args, model_args, data_args = args[0], args[1], args[2]

    exp_args.output_dir = os.path.join(exp_args.output_dir, exp_args.exp_name)

    exp_args.run_name = f"{exp_args.exp_name}__{exp_args.seed}__{int(time.time())}"
    if exp_args.push_to_hub:
        exp_args.run_name = f"{exp_args.run_name}__hub"

        if exp_args.hf_entity is None:
            exp_args.hf_entity = "taresco"
        if exp_args.hf_repo_id is None:
            exp_args.hf_repo_id = f"{exp_args.hf_entity}/{exp_args.exp_name}"
        if exp_args.hf_repo_revision is None:
            exp_args.hf_repo_revision = exp_args.run_name

        exp_args.hf_repo_url = f"https://huggingface.co/{exp_args.hf_repo_id}/tree/{exp_args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if exp_args.with_tracking:
        accelerator_log_kwargs["log_with"] = exp_args.reports_to
        accelerator_log_kwargs["project_dir"] = exp_args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=exp_args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if exp_args.seed:
        set_seed(exp_args.seed)

    # Create output directory in the main process
    if accelerator.is_main_process:
        if exp_args.output_dir is not None:
            os.makedirs(exp_args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Initialize the training dataset
    train_dataset = []
    for i, data_mix in enumerate(data_args.dataset_train):
        logger.info(
            f"Creating training dataset {i + 1} from: {data_mix.get('root_dir', None)}"
        )
        pipeline_mix = data_mix.get("pipeline", None)
        dataset = LocalDataset(
            root_dir=Path(data_mix["root_dir"]),
            pdf_dir_name=data_mix["pdf_dir_name"],
            json_dir_name=data_mix["json_dir_name"],
            pipeline_steps=pipeline_mix,
        )

        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_dataset.append(dataset)

    # Combine all training datasets
    train_dataset = (
        ConcatDataset(train_dataset) if len(train_dataset) > 1 else train_dataset[0]
    )
    train_dataset = train_dataset.filter(
        lambda example: (example["labels"] != -100).any()
    )
    logger.info(f"Total training samples: {len(train_dataset)}")

    # Initialize the evaluation dataset
    data_collator = DataCollator(max_token_len=data_args.max_length)

    eval_dataloaders = {}
    for i, data_mix in enumerate(data_args.dataset_eval):
        logger.info(
            f"Creating eval dataset {i + 1} from: {data_mix.get('root_dir', None)}"
        )
        pipeline_mix = data_mix.get("pipeline", None)
        dataset = LocalDataset(
            root_dir=Path(data_mix["root_dir"]),
            pdf_dir_name=data_mix["pdf_dir_name"],
            json_dir_name=data_mix["json_dir_name"],
            pipeline_steps=pipeline_mix,
        )
        logger.info(f"Found {len(dataset)} samples")

        dl = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=exp_args.per_device_train_batch_size,
        )

        eval_dataloaders[data_mix.get("name", str(i))] = dl

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=exp_args.per_device_train_batch_size,
    )

    # Initialize Model Config if not provided
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You need to specify either a model name or a model configuration name"
        )

    # Initialize the model
    if model_args.model_name_or_path:
        if "Qwen2.5-VL" in model_args.model_name_or_path:
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "Qwen2-VL" in model_args.model_name_or_path:
            model_class = Qwen2VLForConditionalGeneration

        if model_args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.

            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                quantization_config=quantization_config,
                device_map=device_map,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
        else:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
    else:
        logger.info("Training from scratch, no weights or quantization needed")
        model = AutoModel.from_config(config)

    if model_args.use_lora:
        if model_args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=model_args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif model_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Compiling the model if specified
    if model_args.torch_compile:
        logger.info(
            f"Compiling model with torch.compile (backend={model_args.torch_compile_backend}, mode={model_args.torch_compile_mode})"
        )
        model = torch.compile(
            model,
            backend=model_args.torch_compile_backend,
            mode=model_args.torch_compile_mode,
            fullgraph=model_args.torch_compile_fullgraph,
            dynamic=model_args.torch_compile_dynamic,
        )
        logger.info("Model compilation complete")

    # Load the optimizer
    # Set up optimizer
    if exp_args.optimizer == "adamw_torch":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": exp_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=float(exp_args.learning_rate),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {exp_args.optimizer} not supported in custom loop"
        )

    # Total training steps calculation
    samples_per_step = (
        exp_args.per_device_train_batch_size * exp_args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / samples_per_step)
    max_train_steps = int(
        math.ceil(exp_args.num_train_epochs * num_update_steps_per_epoch)
    )
    _max_train_samples = int(math.ceil(exp_args.num_train_epochs * len(train_dataset)))

    # Set up scheduler
    lr_scheduler = get_scheduler(
        name=exp_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * exp_args.warmup_ratio),
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=exp_args.lr_scheduler_kwargs,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_args.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    exp_args.num_train_epochs = math.ceil(
        exp_args.max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = exp_args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if exp_args.with_tracking:
        experiment_config = vars(exp_args)
        dataset_args = vars(data_args)
        modelling_args = vars(model_args)

        all_config = {**experiment_config, **dataset_args, **modelling_args}

        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        if exp_args.wandb_entity is None:
            raise ValueError("Please provide a wandb entity.")

        accelerator.init_trackers(
            exp_args.wandb_project_name,
            all_config,
            init_kwargs={
                "wandb": {
                    "name": exp_args.run_name,
                    "entity": exp_args.wandb_entity,
                    "tags": [exp_args.exp_name],
                }
            },
        )
        _wandb_tracker = accelerator.get_tracker("wandb")

    # Train!
    total_batch_size = (
        exp_args.per_device_train_batch_size
        * accelerator.num_processes
        * exp_args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {exp_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {exp_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {exp_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {exp_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    _progress_bar = tqdm(
        range(exp_args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(exp_args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * exp_args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // exp_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # Evaluate before training
    if accelerator.is_local_main_process:
        metrics = evaluate_model(
            model,
            eval_dataloaders,
            accelerator.device,
            exp_args.reduce_loss,
            "cuda",
            torch.bfloat16,
        )
        logger.info(f"Initial evaluation: {metrics}")
        accelerator.log(metrics, step=completed_steps)

    logger.info(f"Starting from epoch {starting_epoch} and step {completed_steps}.")

    print(data_args)
    print(model_args)


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)
