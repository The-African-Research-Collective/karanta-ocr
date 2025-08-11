import os
import json
import jsonlines
from pathlib import Path

from karanta.data.preprocessing.split_pdf import process_single_pdf

PDF_PATH = "/Users/odunayoogundepo/Desktop/jw/igbo"
MARKDOWN_OUTPUT_DIR = (
    "/Users/odunayoogundepo/newspaper-parser/data/all_data/markdown/jw/igbo"
)
JSON_OUTPUT_DIR = (
    "/Users/odunayoogundepo/newspaper-parser/data/all_data/output_json/jw/igbo"
)
PDF_OUTPUT_DIR = "/Users/odunayoogundepo/newspaper-parser/data/all_data/pdf/jw/igbo"

RESPONSE_FILE_DIR = "/Users/odunayoogundepo/newspaper-parser/data/jw/Igbo/response_dir/batch_llm_requsts_file_allenai_olmOCR_7B_0225_preview_8"

PROCESSED_LOG_FILE = "data/all_data/processed.txt"
FAILED_LOG_FILE = "data/all_data/failed.txt"


# Load the processed log file
def load_processed_log(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as f:
        return set(line.strip() for line in f.readlines())


processed_files = load_processed_log(PROCESSED_LOG_FILE)

with (
    open(PROCESSED_LOG_FILE, "a") as log_file,
    open(FAILED_LOG_FILE, "a") as failed_log_file,
):
    if os.path.isdir(RESPONSE_FILE_DIR):
        for file in os.listdir(os.path.join(RESPONSE_FILE_DIR, "results")):
            if file.endswith(".json"):
                file_path = os.path.join(RESPONSE_FILE_DIR, "results", file)

                pdf_file_name = file.split("-")[0].replace(".pdf", "")
                page_index = file.split("-")[1].split(".")[0]

                new_pdf_file_name = f"{pdf_file_name}_page_{page_index}.pdf"
                new_markdown_file_name = f"{pdf_file_name}_page_{page_index}.md"
                new_json_file_name = f"{pdf_file_name}_page_{page_index}.json"

                whole_pdf_file_name = f"{pdf_file_name}"

                if new_pdf_file_name not in processed_files:
                    try:
                        # check if the new PDF file exists in the PDF_OUTPUT_DIR
                        new_pdf_file_path = os.path.join(
                            PDF_OUTPUT_DIR, new_pdf_file_name
                        )
                        if not os.path.exists(new_pdf_file_path):
                            # locate the whole PDF file
                            whole_pdf_file_path = os.path.join(
                                PDF_PATH, whole_pdf_file_name + ".pdf"
                            )
                            if not os.path.exists(whole_pdf_file_path):
                                print(
                                    f"PDF file {whole_pdf_file_name}.pdf not found in {PDF_PATH}"
                                )
                                continue
                            else:
                                # Split the pdf file into pages
                                print(
                                    f"Processing PDF file: {whole_pdf_file_name} at {whole_pdf_file_path}"
                                )
                                _, status = process_single_pdf(
                                    (whole_pdf_file_path, whole_pdf_file_name),
                                    Path(PDF_OUTPUT_DIR),
                                )

                                if not status:
                                    print(f"Failed to process {whole_pdf_file_name}")
                                    continue
                        else:
                            print(
                                f"PDF file {new_pdf_file_name} already exists in {PDF_OUTPUT_DIR}"
                            )

                        with open(
                            os.path.join(
                                RESPONSE_FILE_DIR,
                                "results",
                                f"{pdf_file_name}.pdf-{str(int(page_index) + 1)}.json",
                            ),
                            "r",
                        ) as f:
                            data = json.load(f)

                        # Write the markdown file
                        markdown_file_path = os.path.join(
                            MARKDOWN_OUTPUT_DIR, new_markdown_file_name
                        )

                        print(f"Writing markdown file: {markdown_file_path}")
                        with open(
                            markdown_file_path, "w", encoding="utf-8"
                        ) as markdown_file:
                            markdown_file.write(
                                json.loads(data["result"]["text"])["natural_text"]
                            )

                        # Write the JSON file
                        json_file_path = os.path.join(
                            JSON_OUTPUT_DIR, new_json_file_name
                        )
                        print(f"Writing JSON file: {json_file_path}")

                        with open(json_file_path, "w", encoding="utf-8") as json_file:
                            json.dump(data, json_file, ensure_ascii=False, indent=2)

                        processed_files.add(new_pdf_file_name)
                        log_file.write(new_pdf_file_name + "\n")
                    except json.decoder.JSONDecodeError as e:
                        print(f"Failed to process {file_path}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
                    except TypeError as e:
                        print(f"Failed to process {file_path}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
                    except FileNotFoundError as e:
                        print(f"Failed to process {file_path}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
    elif os.path.isfile(RESPONSE_FILE_DIR):
        with jsonlines.open(RESPONSE_FILE_DIR) as reader:
            for obj in reader:
                file = obj["request_id"]

                pdf_file_name = "".join(file.split("-")[:-1]).replace(".pdf", "")
                page_index = int(file.split("-")[-1]) - 1

                if page_index < 0:
                    print(f"Skipping invalid page index {page_index} for file {file}")
                    continue

                new_pdf_file_name = f"{pdf_file_name}_page_{page_index}.pdf"
                new_markdown_file_name = f"{pdf_file_name}_page_{page_index}.md"
                new_json_file_name = f"{pdf_file_name}_page_{page_index}.json"

                whole_pdf_file_name = f"{pdf_file_name}"

                if new_pdf_file_name not in processed_files:
                    try:
                        # check if the new PDF file exists in the PDF_OUTPUT_DIR
                        new_pdf_file_path = os.path.join(
                            PDF_OUTPUT_DIR, new_pdf_file_name
                        )
                        if not os.path.exists(new_pdf_file_path):
                            # locate the whole PDF file
                            whole_pdf_file_path = os.path.join(
                                PDF_PATH, whole_pdf_file_name + ".pdf"
                            )
                            if not os.path.exists(whole_pdf_file_path):
                                print(
                                    f"PDF file {whole_pdf_file_name}.pdf not found in {PDF_PATH}"
                                )
                                continue
                            else:
                                # Split the pdf file into pages
                                print(
                                    f"Processing PDF file: {whole_pdf_file_name} at {whole_pdf_file_path}"
                                )
                                _, status = process_single_pdf(
                                    (whole_pdf_file_path, whole_pdf_file_name),
                                    Path(PDF_OUTPUT_DIR),
                                )

                                if not status:
                                    print(f"Failed to process {whole_pdf_file_name}")
                                    continue
                        else:
                            print(
                                f"PDF file {new_pdf_file_name} already exists in {PDF_OUTPUT_DIR}"
                            )

                        data = {
                            "task_id": "".join(obj["request_id"].split("-")[:-1])
                            + str(int(file.split("-")[-1]) - 1),
                            "result": {
                                "text": obj["response"],
                                "finish_reason": "stop",
                                "model": "allenai/olmOCR-7B-0225-preview",
                                "usage": obj["usage"],
                                "metadata": {
                                    "generation_time": obj["response_time"],
                                    "server_url": obj["server_used"],
                                    "generation_params": {
                                        "model": "allenai/olmOCR-7B-0225-preview",
                                        "max_tokens": 6000,
                                        "temperature": 0.1,
                                    },
                                    "timestamp": obj["timestamp"],
                                },
                            },
                        }

                        # Write the markdown file
                        markdown_file_path = os.path.join(
                            MARKDOWN_OUTPUT_DIR, new_markdown_file_name
                        )

                        print(f"Writing markdown file: {markdown_file_path}")
                        with open(
                            markdown_file_path, "w", encoding="utf-8"
                        ) as markdown_file:
                            markdown_file.write(
                                json.loads(data["result"]["text"])["natural_text"]
                            )

                        # Write the JSON file
                        json_file_path = os.path.join(
                            JSON_OUTPUT_DIR, new_json_file_name
                        )
                        print(f"Writing JSON file: {json_file_path}")

                        with open(json_file_path, "w", encoding="utf-8") as json_file:
                            json.dump(data, json_file, ensure_ascii=False, indent=2)

                        processed_files.add(new_pdf_file_name)
                        log_file.write(new_pdf_file_name + "\n")
                    except json.decoder.JSONDecodeError as e:
                        print(f"Failed to process {obj['request_id']}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
                    except TypeError as e:
                        print(f"Failed to process {obj['request_id']}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
                    except FileNotFoundError as e:
                        print(f"Failed to process {obj['request_id']}: {e}")
                        failed_log_file.write(new_pdf_file_name + "\n")
