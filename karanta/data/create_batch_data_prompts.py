import yaml

from jinja2 import Template

from karanta.data.process_pdf_utils import render_pdf_to_base64png
from karanta.prompts.anchor import get_anchor_text

TARGET_IMAGE_DIM = 2048
PROMPT_PATH = "configs/prompts/open_ai_data_generation.yaml"


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


def build_page_query(local_pdf_path: str, pretty_pdf_path: str, page: int) -> dict:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, TARGET_IMAGE_DIM)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    with open(PROMPT_PATH, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        if "system" in prompt_template_dict:
            prompt_template_dict["system"] = Template(prompt_template_dict["system"])

    # DEBUG crappy temporary code here that does the actual api call live so I can debug it a bit
    # from openai import OpenAI

    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # print(
    #     f"Prompt: {prompt_template_dict['system'].render({'base_text': anchor_text})}"
    # )

    # response = client.chat.completions.create(
    #     model="gpt-4.1-mini-2025-04-14",
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
    # print(response)

    # Construct OpenAI Batch API request format#
    # There are a few tricks to know when doing data processing with OpenAI's apis
    # First off, use the batch query system, it's 1/2 the price and exactly the same performance
    # Second off, use structured outputs. If your application is not an actual chatbot, use structured outputs!
    # Even if the last 10 queries you ran with the regular chat api returned exactly what you wanted without extra "LLM fluff text", that doesn't mean this will hold across 1000's of queries
    # Also, structured outputs let you cheat, because the order in which fields are in your schema, is the order in which the model will answer them, so you can have it answer some "preperatory" or "chain of thought" style questions first before going into the meat of your response, which is going to give better answers
    # Check your prompt for typos, it makes a performance difference!
    # Ask for logprobs, it's not any more expensive and you can use them later to help identify problematic responses
    return {
        "custom_id": f"{pretty_pdf_path}-{page}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
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


if __name__ == "__main__":
    local_pdf_path = "/Users/odunayoogundepo/Downloads/Agbeyewo.pdf"
    page_num = 2

    print(build_page_query(local_pdf_path, local_pdf_path, 1))
