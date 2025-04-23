#!/usr/bin/env python
import argparse

from datasets import load_dataset


def main(args: argparse.ArgumentParser):
    dataset = load_dataset(args.dataset_name, split=args.split, private=args.private)
    dataset.push_to_hub(f"taresco/{args.hub_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a dataset and push to hub.")
    parser.add_argument(
        "--dataset_name", required=True, help="Name of dataset on HF hub"
    )
    parser.add_argument(
        "--split",
        help=(
            "Splits to be considered. Defaults to 'train'. "
            "'train[10:20]' From record 10 (included) to record 20 (excluded) of `train` split. "
            "'train[:10%]+train[-80%:]' The first 10% of train + the last 80% of train. "
            "'train+test' combines both splits. More details available at "
            "https://huggingface.co/docs/datasets/v1.2.0/splits.html#slicing-api"
        ),
        default="train",
    )
    parser.add_argument(
        "--hub_id",
        type=str,
        required=True,
        help="Name or ID for dataset on hub",
    )
    parser.add_argument(
        "--private",
        type=bool,
        help="If dataset repo should be made private",
        default=False,
    )
    args = parser.parse_args()
    main(args)
