"""
This script contains functions to create batch data prompts for OpenAI and VLLM models.
For generating training and testing data, it uses the OpenAI API to process PDF files and extract relevant information.

Sample usage:python -m karanta.data.create_batch_data_prompts --data_path /Users/odunayoogundepo/Downloads/annotation_images \
                --output_path /Users/odunayoogundepo/Downloads/annotation_images_transcribed \
                --model_group openai \
                --model gpt-4.1-mini-2025-04-14
"""

import os
import yaml
import json
import logging
import random
import jsonlines

from enum import Enum
from jinja2 import Template
from pypdf import PdfReader

from karanta.data.process_pdf_utils import render_pdf_to_base64png
from karanta.prompts.anchor import get_anchor_text
from karanta.data.utils import timeit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up logger
logger = logging.getLogger(__name__)

TARGET_IMAGE_DIM = 2048
PROMPT_PATH = "configs/prompts/open_ai_data_generation.yaml"


class ModelGroup(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLMO_VLLM = "olm_vllm"


class Model(str, Enum):
    GPT_4_1 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_O = "gpt-4o-2024-08-06"
    OLMO_7B_PREVIEW = "allenai/olmOCR-7B-0225-preview"


def openai_response_format_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "page_response",
            "schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "additionalProperties": False,
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
            "strict": True,
        },
    }


@timeit
def build_page_query_openai(
    local_pdf_path: str, pretty_pdf_path: str, page: int, model_name: str
) -> dict:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, TARGET_IMAGE_DIM)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    with open(PROMPT_PATH, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        # if "system" in prompt_template_dict:
        #     prompt_template_dict["system"] = Template(prompt_template_dict["system"])

        if "newspaper_system" in prompt_template_dict:
            prompt_template_dict["system"] = Template(
                prompt_template_dict["newspaper_system"]
            )

    # DEBUG crappy temporary code here that does the actual api call live so I can debug it a bit
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(
        f"Prompt: {prompt_template_dict['system'].render({'base_text': anchor_text})}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
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
        temperature=0.1,
        max_tokens=3000,
        logprobs=True,
        top_logprobs=5,
        response_format=openai_response_format_schema(),
    )
    print(response.choices[0].message.content)
    print(json.loads(response.choices[0].message.content)["natural_text"])

    # Construct OpenAI Batch API request format#
    # There are a few tricks to know when doing data processing with OpenAI's apis
    # First off, use the batch query system, it's 1/2 the price and exactly the same performance
    # Second off, use structured outputs. If your application is not an actual chatbot, use structured outputs!
    # Even if the last 10 queries you ran with the regular chat api returned exactly what you wanted without extra "LLM fluff text", that doesn't mean this will hold across 1000's of queries
    # Also, structured outputs let you cheat, because the order in which fields are in your schema, is the order in which the model will answer them, so you can have it answer some "preperatory" or "chain of thought" style questions first before going into the meat of your response, which is going to give better answers
    # Check your prompt for typos, it makes a performance difference!
    # Ask for logprobs, it's not any more expensive and you can use them later to help identify problematic responses

    # return {
    #     "custom_id": f"{pretty_pdf_path}-{page}",
    #     "method": "POST",
    #     "url": "/v1/chat/completions",
    #     "body": {
    #         "model": model_name,
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": prompt_template_dict["system"].render(
    #                             {"base_text": anchor_text}
    #                         ),
    #                     },
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/png;base64,{image_base64}"
    #                         },
    #                     },
    #                 ],
    #             }
    #         ],
    #         "temperature": 0.1,
    #         "max_tokens": 6000,
    #         "logprobs": True,
    #         "top_logprobs": 5,
    #         "response_format": openai_response_format_schema(),
    #     },
    # }


@timeit
async def build_page_query_vllm_olmoocr(
    local_pdf_path: str, pretty_pdf_path: str, page: int, model_name: str
) -> dict:
    """
    Turns out that the OlmoOCR model is already pretty good on a lot of our content so the goal is to sort of
    distill the output of the model into a separate model and use that to train a new model that is faster and smaller
    """
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, TARGET_IMAGE_DIM)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    with open(PROMPT_PATH, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        if "system" in prompt_template_dict:
            prompt_template_dict["system"] = Template(prompt_template_dict["system"])

    # DEBUG crappy temporary code here that does the actual api call live so I can debug it a bit
    # from openai import OpenAI

    # client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="token-abc123")

    # print(
    #     f"Prompt: {prompt_template_dict['system'].render({'base_text': anchor_text})}"
    # )

    # response = await client.chat.completions.create(
    #     model="allenai/olmOCR-7B-0225-preview",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": prompt_template_dict["system"].render(
    #                         {"base_text": anchor_text}
    #                     ),
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    #                 },
    #             ],
    #         }
    #     ],
    #     temperature=0.1,
    #     max_tokens=3000,
    #     logprobs=True,
    #     top_logprobs=5,
    #     response_format=openai_response_format_schema(),
    # )

    # print(response.choices[0].message.content)

    # Construct A list of batch requests to send to the VLLM server
    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
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
        # check how many pages are in the pdf
        with open(pdf_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)

            if args.num_pages_per_pdf >= total_pages:
                pages_available = total_pages
                sample_pages = range(total_pages)
            else:
                pages_available = args.num_pages_per_pdf
                sample_pages = random.sample(range(total_pages), args.num_pages_per_pdf)

            logger.info(f"Processing {item} with {total_pages} pages")
            logger.info(f"Processing {pages_available} pages")

            # Sample pages from the PDF
            logger.info(f"Sampled pages: {sample_pages}")

            # process the sampled pages
            for page in sample_pages:
                if args.model_group == ModelGroup.OPENAI:
                    prompt = build_page_query_openai(pdf_path, item, page, args.model)
                elif args.model_group == ModelGroup.OLMO_VLLM:
                    prompt = build_page_query_vllm_olmoocr(
                        pdf_path, item, page, args.model
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


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Create batch data prompts and write them to a file."
    # )
    # parser.add_argument(
    #     "--data_path", required=True, help="Path to the directory containing PDF files."
    # )
    # parser.add_argument(
    #     "--output_path",
    #     required=True,
    #     help="Path to the directory to save output prompts.",
    # )
    # parser.add_argument(
    #     "--num_processes",
    #     type=int,
    #     default=4,
    #     help="Number of processes to use for parallel processing.",
    # )
    # parser.add_argument(
    #     "--model_group",
    #     type=ModelGroup,
    #     choices=[model_group.value for model_group in ModelGroup],
    #     required=True,
    #     help="Model group to use for processing.",
    # )
    # parser.add_argument(
    #     "--model",
    #     type=Model,
    #     choices=[model.value for model in Model],
    #     required=True,
    #     help="Model to use for processing.",
    # )
    # parser.add_argument(
    #     "--num_pages_per_pdf",
    #     type=int,
    #     default=10,
    #     help="Number of pages to process per PDF file.",
    # )
    # parser.add_argument(
    #     "--request_per_batch_file",
    #     type=int,
    #     default=1000,
    #     help="Number of requests to write per batch file.",
    # )
    # args = parser.parse_args()
    # main(args)
    local_pdf_path = "/Users/odunayoogundepo/newspaper-parser/data/train_images/2005november_pg_7.pdf"
    page_num = 1

    build_page_query_openai(
        local_pdf_path, local_pdf_path, page_num, "gpt-4.1-2025-04-14"
    )

    # import asyncio

    # asyncio.run(build_page_query_vllm_olmoocr(local_pdf_path, local_pdf_path, page_num))
