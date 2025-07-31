"""
This scipts procesess PDF files in specified folder, converts them to images and uses a model to classify them based on language and general category.
It generates a JSONL file which it uses with together batch inference.

Sample usage:
     python -m karanta.data.get_pdf_metadata --folder_path /path/to/pdf/folder --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""
import os
import json
import base64
import io
import tempfile
import time
import argparse
from pathlib import Path
from enum import Enum

from pydantic import BaseModel
from pdf2image import convert_from_path
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Using togetherai's batch inference return information on the language contained in pdfs contained in a folder."
    )
    parser.add_argument("--folder_path", help="Path to folder containing PDFs of interest")
    parser.add_argument("--output", default="output", help="baseoutput name to store output files")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="prefered model to use for classification. Model should exist on Together server. Defaults to meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    return parser.parse_args()

class DocumentCategory(str, Enum):
    RELIGIOUS = "religious"
    NOVEL = "novel"
    BROCHURE = "brochure"
    NEWSPAPER = "newspaper"
    TEXTBOOK = "textbook"
    ACADEMIC = "academic"
    WEBPAGE = "webpage"
    OTHER = "other"

class DocumentClassification(BaseModel):
    category: DocumentCategory
    language: str
    confidence: float

schema = DocumentClassification.model_json_schema()

def convert_pdf2image(data_path: Path, output_dir: Path):
    """
    Convert PDF files to images using pdf2image.
    """
    return convert_from_path(data_path, output_folder=output_dir, last_page=3, fmt="jpg")

def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def process_pdf_get_max_3_images(pdf_path: str, model: str):
    """Process a single PDF file and return JSONL line with up to 3 images."""
    full_path, item = pdf_path
    pdf_basename = os.path.splitext(item)[0]

    base_data = {
        "custom_id": pdf_basename,
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "These are images from a document. "
                                "Please classify it into one of the following categories "
                                "that best summarizes the overall tone of the document: "
                                "religious, novel, academic, brochure, textbook, webpage, "
                                "newspaper, or other. Also determine the primary language "
                                "of the document based on the images and your confidence in the "
                                "classification (0-1)."
                            )
                        }
                    ]
                }
            ],
            "response_format":{
                "type": "json_schema",
                "schema": schema,
            },  
        }
    }

    with tempfile.TemporaryDirectory() as path:
        images = convert_pdf2image(data_path=full_path, output_dir=path,)
        image_content = []
        
        for i, image in enumerate(images[:3]):  # limit to 3 images
            encoded_image = encode_image(image)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        
        base_data["body"]["messages"][0]["content"].extend(image_content)

    return json.dumps(base_data)

def generate_jsonl_split_by_size(folder_path: str, model: str, base_output_name: str="output", max_file_size_mb: int=100):
    pdf_files = []
    file_list = []

    if os.path.isfile(folder_path):
        pdf_files.append((folder_path, os.path.basename(folder_path)))
    else:
        for item in os.listdir(folder_path):
            full_path = os.path.join(folder_path, item)
            if os.path.isfile(full_path) and item.lower().endswith(".pdf"):
                pdf_files.append((full_path, item))

    max_file_size = max_file_size_mb * 1024 * 1024
    file_index = 1
    current_file_size = 0

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
    f = open(output_file, "w", encoding="utf-8")
    file_list.append(str(output_file))

    for _, pdf_file in enumerate(pdf_files):
        json_line = process_pdf_get_max_3_images(pdf_file, model)
        line_bytes = (json_line + "\n").encode("utf-8")
        line_size = len(line_bytes)

        if current_file_size + line_size > max_file_size:
            f.close()
            file_index += 1
            output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
            f = open(output_file, "w", encoding="utf-8")
            file_list.append(str(output_file))
            current_file_size = 0

        f.write(json_line + "\n")
        current_file_size += line_size

    f.close()
    print(f"Completed. Generated {file_index} JSONL file(s).")
    return file_list

def batch_call(batch_file_path: str):
    file_resp = client.files.upload(file=batch_file_path, purpose="batch-api")
    batch = client.batches.create_batch(file_resp.id, endpoint="/v1/chat/completions")
    base_name =  os.path.splitext(batch_file_path)[0]
    output_file = f"results_{base_name}.jsonl"

    while True:
        batch_status = client.batches.get_batch(batch.id)
        if batch_status.status == "COMPLETED":
            client.files.retrieve_content(id=batch_status.output_file_id, output=output_file)
            break
        elif batch_status.status == "FAILED":
            raise ValueError(f"Batch failed: {batch_status.error}")
        time.sleep(10)  # wait 10s before next check

def main():
    args= parse_args()
    file_list = generate_jsonl_split_by_size(folder_path=args.folder_path, base_output_name=args.output, model=args.model)
    for file in file_list:
        batch_call(file)
    

if __name__ == "__main__":
    main()