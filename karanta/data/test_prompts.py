"""
This file is used to test the model inference against a VLLM server or OPenAI model for a given PDF file to inspect its output.

Sample usage:python -m karanta.data.test_prompts --local_pdf_path /Users/odunayoogundepo/Downloads/post_training_math_v1.pdf \
                --inference_type olmoocr \
                --page_num 1
"""

import os
import yaml
import json
import argparse

from jinja2 import Template
from openai import OpenAI
from dotenv import load_dotenv

from karanta.data.process_pdf_utils import render_pdf_to_base64png
from karanta.prompts.anchor import get_anchor_text
from karanta.data.utils import openai_response_format_schema
from karanta.constants import TARGET_IMAGE_DIM, PROMPT_PATH, Model

load_dotenv()


def test_build_page_query_openai(
    local_pdf_path: str, page: int, model_name: str
) -> dict:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, TARGET_IMAGE_DIM)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    image_page = True

    if len(anchor_text.split("\n")) > 10:
        image_page = False

    with open(PROMPT_PATH, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        if image_page:
            prompt_template_dict["system"] = Template(
                prompt_template_dict["newspaper_system"]
            )
        else:
            prompt_template_dict["system"] = Template(prompt_template_dict["system"])

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(
        f"Prompt: {prompt_template_dict['system'].render({'base_text': anchor_text})}\n========================================================================"
    )

    response = client.chat.completions.create(
        model=model_name,
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
        max_tokens=6000,
        logprobs=True,
        top_logprobs=5,
        response_format=openai_response_format_schema(),
    )
    print(
        f"Response: {response.choices[0].message.content}\n========================================================================"
    )
    print(
        f"Generated Natural Text: {json.loads(response.choices[0].message.content)['natural_text']}"
    )


def test_build_page_query_vllm_olmoocr(
    local_pdf_path: str, page: int, model_name: str, host: str
) -> dict:
    """
    Turns out that the OlmoOCR model is already pretty good on a lot of our content so the goal is to sort of
    distill the output of the model into a separate model and use that to train a new model that is faster and smaller
    """
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, 2048)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    with open(PROMPT_PATH, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        if "olmo_ocr_system_prompt" in prompt_template_dict:
            prompt_template_dict["system"] = Template(
                prompt_template_dict["olmo_ocr_system_prompt"]
            )

    client = OpenAI(base_url=host, api_key="token-abc123")

    print(
        f"Prompt: {prompt_template_dict['system'].render({'base_text': anchor_text})}"
    )

    response = client.chat.completions.create(
        model=model_name,
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

    print(
        f"Response: {response.choices[0].message.content}\n========================================================================"
    )
    print(
        f"Generated Natural Text: {json.loads(response.choices[0].message.content)['natural_text']}"
    )


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
        "--inference_type",
        choices=["openai", "olmoocr"],
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
        )
    elif args.inference_type == "olmoocr":
        test_build_page_query_vllm_olmoocr(
            local_pdf_path=args.local_pdf_path,
            page=args.page_num,
            model_name=args.model_name,
            host=args.host,
        )
