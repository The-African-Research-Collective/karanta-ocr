import logging
import base64
import json
import yaml
import numpy as np

from dataclasses import dataclass
from abc import ABC
from jinja2 import Template

from PIL import Image
from io import BytesIO
from transformers import AutoProcessor
from typing import Any

from karanta.training.utils import SingleDatapoint
from karanta.prompts.anchor import get_anchor_text
from karanta.data.process_pdf_utils import render_pdf_to_base64png

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BasePipelineStep(ABC):
    """This is the base class for all the pipeline steps."""

    def __call__(self, *args, **kwargs):
        """Call the step with the given arguments."""
        pass


@dataclass(frozen=True, slots=True)
class PDF2ImageStep(BasePipelineStep):
    """Pipeline step that renders PDF to image."""

    target_longest_image_dim: int

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """Render PDF to image."""
        print(sample)
        # # Render PDF to image
        base64_png = render_pdf_to_base64png(
            str(sample.pdf_path),
            page_num=1,
            target_longest_image_dim=self.target_longest_image_dim,
        )
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))

        # # Update sample
        sample.image = image

        return sample


@dataclass(frozen=True, slots=True)
class JSONOutputFormat(BasePipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        page_data = sample.page_data
        sample.response = json.dumps(
            {
                "primary_language": page_data["primary_language"],
                "is_rotation_valid": page_data["is_rotation_valid"],
                "rotation_correction": page_data["rotation_correction"],
                "is_table": page_data["is_table"],
                "is_diagram": page_data["is_diagram"],
                "natural_text": page_data["natural_text"],
            },
            ensure_ascii=False,
        )
        return sample


@dataclass(frozen=True, slots=True)
class FetchPageData(BasePipelineStep):
    """Fetch the page data from the JSON file and store it in the sample."""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        with open(sample.json_path, "r", encoding="utf-8") as f:
            output_result = json.loads(json.loads(f.read())["result"]["text"])

        sample.page_data = output_result
        return sample


@dataclass(frozen=True, slots=True)
class StaticLengthDocumentAnchoring(BasePipelineStep):
    target_anchor_text_len: int

    """Pipeline step that runs document anchoring on the PDF and puts in the data to be used by later prompting stages"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        anchor_text = get_anchor_text(
            str(sample.pdf_path),
            page=1,
            pdf_engine="pdfreport",
            target_length=self.target_anchor_text_len,
        )
        sample.anchor_text = anchor_text
        return sample


@dataclass(frozen=True, slots=True)
class FinetuningPrompt(BasePipelineStep):
    """Applies the standard fine tuning prompt"""

    prompt_path: str

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        image_page = True

        if len(sample.anchor_text.split("\n")) > 10:
            image_page = False

        with open(self.prompt_path, "r") as stream:
            prompt_template_dict = yaml.safe_load(stream)

            if image_page:
                prompt_template_dict["system"] = Template(
                    prompt_template_dict["newspaper_system"]
                )
            else:
                prompt_template_dict["system"] = Template(
                    prompt_template_dict["system"]
                )

        sample.instruction_prompt = prompt_template_dict["system"].render(
            {"base_text": sample.anchor_text}
        )
        return sample


@dataclass(frozen=True, slots=True)
class InstructUserMessages(BasePipelineStep):
    """Creates instruction-following messages format for training."""

    prompt_first: bool = False

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        # Prepare messages
        if self.prompt_first:
            messages = {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.instruction_prompt},
                    {"type": "image", "image": sample.image},
                ],
            }
        else:
            messages = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": sample.instruction_prompt},
                ],
            }

        sample.user_messages = messages

        return sample


@dataclass(frozen=True, slots=True)
class Tokenizer(BasePipelineStep):
    """Tokenizes messages and creates training labels with proper masking."""

    processor: Any  # The model processor (e.g., AutoProcessor)
    masking_index: int = -100
    end_of_message_token: str = "<|im_end|>"  # Configurable, defaults to Qwen format

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """Tokenize messages and create labels for training."""

        if isinstance(self.processor, str):
            processor = AutoProcessor.from_pretrained(self.processor)

        # Extract user message and response
        user_messages = sample.user_messages
        response = sample.response

        # Apply chat template to user message only with generation prompt
        # user_messages is a single dict, so wrap it in a list
        text = processor.apply_chat_template(
            [user_messages], tokenize=False, add_generation_prompt=True
        )

        main_image = None
        for usg_msg in user_messages["content"]:
            if "image" in usg_msg:
                main_image = usg_msg["image"]
                break

        assert main_image is not None

        # Process inputs using processor
        inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="np",
        )

        # Get labels by tokenizing the output text
        labels = processor(text=[response], padding=True, return_tensors="np")

        # Append end-of-message token to the labels
        end_tokens = processor.tokenizer(
            self.end_of_message_token, add_special_tokens=False
        )["input_ids"]
        end_tokens = np.array(end_tokens, dtype=inputs.input_ids.dtype)

        # Handle the case where labels['input_ids'] is empty
        if labels["input_ids"].shape[1] == 0:
            labels_input_ids_0 = np.array([], dtype=inputs.input_ids.dtype)
        else:
            labels_input_ids_0 = labels["input_ids"][0].astype(inputs.input_ids.dtype)

        labels["input_ids"] = np.concatenate([labels_input_ids_0, end_tokens])
        labels["input_ids"] = np.expand_dims(labels["input_ids"], axis=0)

        # Concatenate input_ids and labels
        input_ids = np.concatenate([inputs.input_ids[0], labels.input_ids[0]], axis=0)

        # All columns will participate in attention fully
        attention_mask = np.ones_like(input_ids)

        # Create labels, masking the input portion with -100
        labels_full = np.full_like(input_ids, fill_value=self.masking_index)
        labels_full[len(inputs.input_ids[0]) :] = labels.input_ids[0]

        # Return as dict, including pixel_values
        sample.model_inputs["input_ids"] = input_ids
        sample.model_inputs["attention_mask"] = attention_mask
        sample.model_inputs["labels"] = labels_full
        sample.model_inputs["pixel_values"] = inputs.pixel_values

        if hasattr(inputs, "image_grid_thw"):
            sample.model_inputs["image_grid_thw"] = inputs.image_grid_thw[0]

        return sample
