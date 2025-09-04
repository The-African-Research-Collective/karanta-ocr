"""
This script contains functions to create batch data prompts for OpenAI and VLLM models.
For generating training and testing data, it uses the OpenAI API to process PDF files and extract relevant information.

Sample usage:python -m karanta.data.create_batch_data_prompts --data_path /Users/odunayoogundepo/Downloads/annotation_images \
                --output_path /Users/odunayoogundepo/Downloads/annotation_images_transcribed \
                --model_group openai \
                --model gpt-4.1-mini-2025-04-14 \
                --azure_source_dir ...
"""

import os
import logging
import random
import argparse
import jsonlines

from pypdf import PdfReader

from karanta.data.utils import (
    timeit,
    openai_response_format_schema,
    prepare_image_and_text,
    load_prompt_template,
)
from karanta.constants import TARGET_IMAGE_DIM, ModelGroup, Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up logger
logger = logging.getLogger(__name__)


def convert_img_to_gray(image):
    """Convert image to grayscale."""
    if image.mode != "L":
        image = image.convert("L")
    return image


@timeit
def build_page_query_openai(
    local_pdf_path: str,
    page: int,
    model_name: str,
    azure_source_dir: str,
    convert_to_grayscale: bool = True,
    prompt_key: str = "olmo_ocr_system_prompt",
) -> dict:
    image_base64, anchor_text = prepare_image_and_text(
        local_pdf_path,
        page,
        target_dim=TARGET_IMAGE_DIM,
        convert_to_grayscale=convert_to_grayscale,
    )
    prompt_template_dict = load_prompt_template(prompt_key)
    pretty_pdf_path = os.path.basename(local_pdf_path)

    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
        "azure_source_dir": azure_source_dir,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_template_dict["system"].render(
                                {"base_text": anchor_text}
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 6000,
            "logprobs": True,
            "top_logprobs": 5,
            "response_format": openai_response_format_schema(),
        },
    }


@timeit
def build_page_query_vllm_olmoocr(
    local_pdf_path: str,
    page: int,
    model_name: str,
    azure_source_dir: str = None,
    convert_to_grayscale: bool = True,
    prompt_key: str = "olmo_ocr_system_prompt",
) -> dict:
    """
    Turns out that the OlmoOCR model is already pretty good on a lot of our content so the goal is to sort of
    distill the output of the model into a separate model and use that to train a new model that is faster and smaller
    """
    image_base64, anchor_text = prepare_image_and_text(
        local_pdf_path,
        page,
        target_dim=TARGET_IMAGE_DIM,
        convert_to_grayscale=convert_to_grayscale,
    )
    prompt_template_dict = load_prompt_template(prompt_key)
    pretty_pdf_path = os.path.basename(local_pdf_path)

    # Construct A list of batch requests to send to the VLLM server
    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
        "azure_source_dir": azure_source_dir,
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template_dict["system"].render(
                            {"base_text": anchor_text}
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 6000,
        "logprobs": True,
        "top_logprobs": 5,
        "response_format": openai_response_format_schema(),
    }


def main(args):
    logger.info("Starting batch data prompt generation...")

    # Collect all PDF files to process
    pdf_files = []
    if os.path.isfile(args.data_path):
        # If a single file is provided, add it to the list
        pdf_files.append((args.data_path, os.path.basename(args.data_path)))
    else:
        for item in os.listdir(args.data_path):
            full_path = os.path.join(args.data_path, item)
            if os.path.isfile(full_path) and item.lower().endswith(".pdf"):
                pdf_files.append((full_path, item))

    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Now process the each pdf file
    file_count = 0
    write_count = 0
    output_file_path = os.path.join(
        args.output_path,
        f"batch_llm_requsts_file_{args.model.replace('/', '_').replace('.', '_').replace('-', '_')}_{file_count}.jsonl",
    )
    output_file = jsonlines.open(output_file_path, "a")

    for pdf_path, item in pdf_files:
        try:
            # check how many pages are in the pdf
            with open(pdf_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                total_pages = len(reader.pages)

                if args.num_pages_per_pdf == -1:
                    pages_available = total_pages
                    sample_pages = range(total_pages)
                elif args.num_pages_per_pdf >= total_pages:
                    pages_available = total_pages
                    sample_pages = range(total_pages)
                else:
                    pages_available = args.num_pages_per_pdf
                    sample_pages = random.sample(
                        range(total_pages), args.num_pages_per_pdf
                    )

                logger.info(f"Processing {item} with {total_pages} pages")
                logger.info(f"Processing {pages_available} pages")

                # Sample pages from the PDF
                logger.info(f"Sampled pages: {sample_pages}")

                # process the sampled pages
                for page in sample_pages:
                    if args.model_group == ModelGroup.OPENAI:
                        prompt = build_page_query_openai(
                            pdf_path,
                            page,
                            args.model,
                            azure_source_dir=args.azure_source_dir,
                        )
                    elif args.model_group == ModelGroup.OLMO_VLLM:
                        prompt = build_page_query_vllm_olmoocr(
                            pdf_path,
                            page,
                            args.model,
                            azure_source_dir=args.azure_source_dir,
                        )
                    else:
                        raise ValueError(f"Unsupported model group: {args.model_group}")

                    # Save the prompt to a file
                    output_file.write(prompt)
                    write_count += 1

                    # Check if we need to create a new file
                    if write_count >= args.request_per_batch_file:
                        output_file.close()
                        file_count += 1
                        write_count = 0
                        output_file_path = os.path.join(
                            args.output_path,
                            f"batch_llm_requsts_file_{args.model.replace('/', '_').replace('.', '_').replace('-', '_')}_{file_count}.jsonl",
                        )
                        output_file = jsonlines.open(output_file_path, "a")
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create batch data prompts and write them to a file."
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to the directory containing PDF files."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the directory to save output prompts.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument(
        "--model_group",
        type=ModelGroup,
        choices=[model_group.value for model_group in ModelGroup],
        required=True,
        help="Model group to use for processing.",
    )
    parser.add_argument(
        "--model",
        type=Model,
        choices=[model.value for model in Model],
        required=True,
        help="Model to use for processing.",
    )
    parser.add_argument(
        "--num_pages_per_pdf",
        type=int,
        default=-1,
        help="Number of pages to process per PDF file.",
    )
    parser.add_argument(
        "--request_per_batch_file",
        type=int,
        default=1000,
        help="Number of requests to write per batch file.",
    )
    parser.add_argument(
        "--azure_source_dir",
        type=str,
        default=None,
        required=True,
        help="Azure source directory for the prompts.",
    )
    args = parser.parse_args()
    main(args)
