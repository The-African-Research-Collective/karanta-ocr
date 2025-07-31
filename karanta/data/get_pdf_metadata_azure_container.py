"""
This script processes PDF files in an Azure blob container,
converts them to images, and generates JSONL files for TogetherAI's batch inference.
It then asynchronously submits JSONL files to TogetherAI as soon as they are ready.

Sample usage:
     python -m karanta.data.get_pdf_metadata --container_name mycontainer --output my_output --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
"""

import os
import json
import base64
import io
import asyncio
from asyncio import Semaphore
import argparse
from pathlib import Path
from enum import Enum
from typing import Set

from pydantic import BaseModel
from pdf2image import convert_from_bytes
from together import Together
from dotenv import load_dotenv

from azure.storage.blob.aio import ContainerClient
from azure.core.pipeline.transport import AioHttpTransport

transport = AioHttpTransport(read_timeout=600) 
load_dotenv()

################################### ACCESS AZURE ##################################
account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
sas_token = os.getenv("AZURE_SAS_TOKEN")
# ###################################################################################

client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Using TogetherAI's batch inference return information on the language contained in PDFs in a blob container."
    )
    parser.add_argument("--container_name", help="Container name in Azure blob containing PDFs of interest")
    parser.add_argument("--output", default="output", help="Base output name to store output files")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="Preferred TogetherAI model for classification. Please see avaialble models at https://docs.together.ai/docs/batch-inference#model-availability")
    parser.add_argument("--max_concurrent_pdfs", type=int, default=1, help="Maximum number of concurrent PDF processing tasks (default 1)")
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

def convert_pdf2image(pdf_file: bytes):
    """Convert PDF files to images using pdf2image."""
    return convert_from_bytes(pdf_file, last_page=3, fmt="jpg")

def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def process_pdf(pdf_file: bytes, blob_name: str, model: str):
    """Process a single PDF file and return JSONL line with up to 3 images."""
    print(f"Processing file:{blob_name} ...")

    base_data = {
        "custom_id": blob_name,
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

    images = convert_pdf2image(pdf_file)
    image_content = []
        
    for i, image in enumerate(images[:3]):  # limit to 3 images
        encoded_image = encode_image(image)
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
        
    base_data["body"]["messages"][0]["content"].extend(image_content)
    return json.dumps(base_data)

async def process_blob_pdf(container: ContainerClient, blob_name:str, model: str, sem: Semaphore, queue: asyncio.Queue):
    async with sem:
        # Download PDF to memory
        stream = await container.download_blob(blob_name)
        pdf_file = await stream.readall()

        # Convert to JSONL line
        json_line = process_pdf(pdf_file, blob_name, model)

        # Send line to writer queue
        await queue.put(json_line)

async def jsonl_writer(queue: asyncio.Queue, completed_files_queue: asyncio.Queue, base_output_name: str, max_file_size_mb: int = 100):
    """Consumes JSON lines from queue and writes to rotated JSONL files,
    and signals completed files for batch submission."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    max_file_size = max_file_size_mb * 1024 * 1024
    file_index = 1
    current_file_size = 0
    output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
    f = open(output_file, "w", encoding="utf-8")
    print(f"Writing to {output_file}")

    while True:
        json_line = await queue.get()
        if json_line is None:  # Poison pill
            break

        line_bytes = (json_line + "\n").encode("utf-8")
        line_size = len(line_bytes)

        # Rotate file if needed
        if current_file_size + line_size > max_file_size:
            f.close()
            await completed_files_queue.put(str(output_file))  # Notify submitter

            file_index += 1
            output_file = output_dir / f"{base_output_name}_{file_index}.jsonl"
            f = open(output_file, "w", encoding="utf-8")
            print(f"Rotating to {output_file}")
            current_file_size = 0

        f.write(json_line + "\n")
        current_file_size += line_size

    f.close()
    await completed_files_queue.put(str(output_file))
    print(f"✅ Completed writing {file_index} JSONL file(s).")

async def batch_call(batch_file_path: str):
    """Submits a JSONL file to TogetherAI's batch API and waits for completion."""
    file_resp = client.files.upload(file=batch_file_path, purpose="batch-api")
    batch = client.batches.create_batch(file_resp.id, endpoint="/v1/chat/completions")
    base_name = os.path.splitext(batch_file_path)[0]
    output_file = f"results_{base_name}.jsonl"

    while True:
        batch_status = client.batches.get_batch(batch.id)
        if batch_status.status == "COMPLETED":
            client.files.retrieve_content(id=batch_status.output_file_id, output=output_file)
            break
        elif batch_status.status == "FAILED":
            raise ValueError(f"Batch failed: {batch_status.error}")
        await asyncio.sleep(10)  # async sleep

async def batch_submitter(completed_files_queue: asyncio.Queue):
    """Manages batch submissions of batch_call"""
    submitted_files: Set[str] = set()

    while True:
        file_path = await completed_files_queue.get()
        if file_path is None:  # Poison pill
            break

        if file_path in submitted_files:
            continue

        print(f"🚀 Submitting {file_path} to TogetherAI batch API...")
        submitted_files.add(file_path)
        await batch_call(file_path)
        print(f"✅ Completed batch submission for {file_path}")

async def concurrent_stream_process(container_name: str, model: str, max_concurrent_pdfs: int, output: str):
    sem = Semaphore(max_concurrent_pdfs)
    queue = asyncio.Queue()
    completed_files_queue = asyncio.Queue()

    container = ContainerClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        container_name=container_name,
        credential=sas_token,
        transport=transport
    )

    writer_task = asyncio.create_task(jsonl_writer(queue, completed_files_queue, output, 100))
    submitter_task = asyncio.create_task(batch_submitter(completed_files_queue))

    async with container:
        tasks = []
        async for blob in container.list_blobs():
            if blob.name.lower().endswith(".pdf"):
                tasks.append(asyncio.create_task(process_blob_pdf(container, blob.name, model, sem, queue)))

        # Wait for all processing to finish
        await asyncio.gather(*tasks)

    # Signal writer to stop
    await queue.put(None)
    await writer_task

    # Signal batch submitter to stop
    await completed_files_queue.put(None)
    await submitter_task

def main():
    args = parse_args()
    asyncio.run(concurrent_stream_process(
        container_name=args.container_name,
        model=args.model,
        max_concurrent_pdfs=args.max_concurrent_pdfs,
        output=args.output
    ))

if __name__ == "__main__":
    main()
