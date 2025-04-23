from pathlib import Path

from PIL import Image

from pdf2image import convert_from_path
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
check_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
check_model.to(device)

# processor = 
# classify_model =
# classify_model.to(device)

def check_for_text_in_image(image:Image.Image):
    prompt = "[INST] <image>\n Answer Yes or No, does this image contain text? [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = check_model.generate(**inputs, max_new_tokens=100)
    response = processor.decode(output[0], skip_special_tokens=True)

    return True if response.lower() == "yes" else False

def check_if_image_requires_segmentation(image:Image.Image):
    # pass image to fine-tuned VLLM to determine label
    pass

def split_pdf2image_and_add_to_dataframe(data_path: Path, output_dir: Path):
    return convert_from_path(data_path, output_folder=output_dir)
