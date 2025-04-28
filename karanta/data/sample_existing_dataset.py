#!/usr/bin/env python
import json
import argparse

from datasets import load_dataset


def main(args: argparse.ArgumentParser):
    parent_dataset = load_dataset(
        args.dataset_name, split=args.split, private=args.private
    )
    column_names = parent_dataset.column_names
    rename_columns = json.loads(args.rename_columns) if args.rename_columns else None
    renamed_columns = []

    if rename_columns:
        """Rename columns if provided"""
        for key, value in rename_columns.items():
            parent_dataset = parent_dataset.map(
                lambda example: {value: example[key]}, remove_columns=[key]
            )
            renamed_columns.append(value)

    if args.columns_to_keep:
        """Keep only the specified columns"""
        columns_to_keep = args.columns_to_keep.split(",")
        columns_to_keep.extend(renamed_columns)
        columns_to_remove = list(set(column_names) - set(columns_to_keep))
        dataset = parent_dataset.map(remove_columns=columns_to_remove)
    else:
        dataset = parent_dataset

    dataset.push_to_hub(f"taresco/{args.hub_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a dataset and push to hub.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=(
            "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        ),
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
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
        "__columns_to_keep",
        type=str,
        help="Comma separated list of columns to keep",
        default=None,
    )
    parser.add_argument(
        "--rename_columns",
        type=str,  # Accept as a JSON string
        help=(
            "If you want to rename columns, provide a JSON string. "
            'Example: \'{"old_name": "new_name", "old_name2": "new_name2"}\''
        ),
        default=None,
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
        help="If new dataset repo should be made private",
        default=False,
    )
    args = parser.parse_args()
    main(args)
