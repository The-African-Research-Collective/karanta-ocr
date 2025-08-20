import os
import time
import logging
import transformers

from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration

from karanta.training.utils import ArgumentParserPlus
from karanta.training.ocr_training_args import (
    ExperimentArguments,
    ModelArguments,
    DatasetArguments,
)

logger = get_logger(__name__)


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

    print(data_args)
    print(model_args)


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)
