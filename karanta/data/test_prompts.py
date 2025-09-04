"""
This file is used to test the model inference against a VLLM server or OPenAI model for a given PDF file to inspect its output.

Sample usage:python -m karanta.data.test_prompts --local_pdf_path /Users/odunayoogundepo/Downloads/post_training_math_v1.pdf \
                --inference_type olmoocr \
                --page_num 1
"""

import os
import argparse

from openai import OpenAI
from dotenv import load_dotenv

from karanta.data.utils import (
    openai_response_format_schema,
    prepare_image_and_text,
    load_prompt_template,
    create_vision_message,
    print_results,
)
from karanta.constants import TARGET_IMAGE_DIM, Model
from karanta.llm_clients.azure_client import AzureOPENAILLM

load_dotenv()


# Refactored main functions
async def test_build_page_query_azure(
    local_pdf_path: str,
    page: int,
    model_name: str,
    convert_to_grayscale: bool = True,
    prompt_key: str = "olmo_ocr_system_prompt",
) -> dict:
    """Test function for Azure OpenAI API."""
    # Prepare common data
    image_base64, anchor_text = prepare_image_and_text(
        local_pdf_path, page, convert_to_grayscale=convert_to_grayscale
    )
    prompt_template = load_prompt_template(prompt_key)

    # Create client and message
    client = AzureOPENAILLM("gpt-4o")
    messages = create_vision_message(prompt_template, anchor_text, image_base64)

    # Make API call
    response = await client.completion(
        [messages],  # Azure client expects nested structure
        openai_response_format_schema(),
        temperature=0.1,
        max_tokens=6000,
    )

    # Print results
    response_content = response[0].generation
    print_results(prompt_template, anchor_text, str(response_content))

    if isinstance(response_content, dict) and "natural_text" in response_content:
        print(f"Generated Natural Text: {response_content['natural_text']}")


def test_build_page_query_openai(
    local_pdf_path: str,
    page: int,
    model_name: str,
    convert_to_grayscale: bool = True,
    prompt_key: str = "olmo_ocr_system_prompt",
) -> dict:
    """Test function for OpenAI API."""
    # Prepare common data
    image_base64, anchor_text = prepare_image_and_text(
        local_pdf_path, page, convert_to_grayscale=convert_to_grayscale
    )
    prompt_template = load_prompt_template(prompt_key)

    # Create client and make API call
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = create_vision_message(prompt_template, anchor_text, image_base64)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=6000,
        logprobs=True,
        top_logprobs=5,
        response_format=openai_response_format_schema(),
    )

    # Print results
    response_content = response.choices[0].message.content
    print_results(prompt_template, anchor_text, response_content)


def test_build_page_query_vllm_olmoocr(
    local_pdf_path: str,
    page: int,
    model_name: str,
    host: str,
    convert_to_grayscale: bool = True,
    prompt_key: str = "olmo_ocr_system_prompt",
) -> dict:
    """
    Test function for VLLM OlmoOCR model.

    Turns out that the OlmoOCR model is already pretty good on a lot of our content so the goal is to sort of
    distill the output of the model into a separate model and use that to train a new model that is faster and smaller
    """
    # Prepare common data (uses 2048 for this model)
    image_base64, anchor_text = prepare_image_and_text(
        local_pdf_path,
        page,
        target_dim=TARGET_IMAGE_DIM,
        convert_to_grayscale=convert_to_grayscale,
    )
    prompt_template = load_prompt_template(prompt_key)

    # Create client and make API call
    client = OpenAI(base_url=host, api_key="token-abc123")
    messages = create_vision_message(prompt_template, anchor_text, image_base64)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=3000,  # Different max_tokens for this model
        logprobs=True,
        top_logprobs=5,
        response_format=openai_response_format_schema(),
    )

    # Print results
    response_content = response.choices[0].message.content
    print_results(prompt_template, anchor_text, response_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenAI page query builder.")
    parser.add_argument(
        "--local_pdf_path",
        required=True,
        help="Path to the local PDF file to process.",
    )
    parser.add_argument(
        "--page_num",
        type=int,
        default=1,
        help="Page number to process from the PDF file.",
    )
    parser.add_argument(
        "--convert_to_grayscale",
        action="store_true",
        help="Convert the image to grayscale before sending to the model.",
    )
    parser.add_argument(
        "--inference_type",
        choices=["openai", "olmoocr", "azure"],
        required=True,
        help="Type of inference to perform: 'openai' for OpenAI models or 'olmoocr' for OlmoOCR.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=Model.OLMO_7B_0725.value,
        help="Name of the model to use for inference. Defaults to OLMO 7B 0725.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8005/v1",
        help="Host URL for the inference service. Defaults to http://localhost:8005/v1.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.local_pdf_path), (
        f"PDF file does not exist: {args.local_pdf_path}"
    )
    assert args.model_name in [model.value for model in Model], (
        f"Invalid model name: {args.model_name}. Available models: {[model.value for model in Model]}"
    )

    if args.inference_type == "openai":
        test_build_page_query_openai(
            local_pdf_path=args.local_pdf_path,
            page=args.page_num,
            model_name=args.model_name,
            convert_to_grayscale=args.convert_to_grayscale,
        )
    elif args.inference_type == "olmoocr":
        test_build_page_query_vllm_olmoocr(
            local_pdf_path=args.local_pdf_path,
            page=args.page_num,
            model_name=args.model_name,
            host=args.host,
            convert_to_grayscale=args.convert_to_grayscale,
        )
    elif args.inference_type == "azure":
        import asyncio

        asyncio.run(
            test_build_page_query_azure(
                local_pdf_path=args.local_pdf_path,
                page=args.page_num,
                model_name=args.model_name,
                convert_to_grayscale=args.convert_to_grayscale,
            )
        )
