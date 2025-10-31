#!/usr/bin/env python3

import argparse
import datetime
import json
from pathlib import Path
from multiprocessing import Pool
import internetarchive

# Map ISO codes or short names to full Archive.org language names
LANGUAGE_MAP = {
    "yor": "Yoruba",
    "hau": "Hausa",
    "igb": "Igbo",
    "swa": "Swahili",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "amh": "Amharic",
    "tsn": "Tswana",
    "zul": "Zulu",
    "xho": "Xhosa",
    "sn": "Shona",
    "wol": "Wolof",
    "tiv": "Tiv",
    "twi": "Twi",
    "eng": "English",
    "fra": "French",
    "ara": "Arabic",
    "por": "Portuguese",
    "som": "Somali",
    "tigr": "Tigrinya",
    "run": "Kirundi",
}


def resolve_language_name(input_lang: str) -> str:
    normalized = input_lang.strip().lower()

    # Try direct code match
    if normalized in LANGUAGE_MAP:
        return LANGUAGE_MAP[normalized]

    # Try reverse match
    for code, name in LANGUAGE_MAP.items():
        if normalized == name.lower():
            return name

    # Fallback
    return input_lang.strip().title()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download PDFs from Internet Archive filtered by exact language."
    )
    parser.add_argument(
        "language", help="Language name or code (e.g., yor, English, Swahili)"
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory to store PDFs (default: ./output)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel downloads"
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Include derived PDFs if no original PDFs found (default: off)",
    )
    return parser.parse_args()


def create_log_entry(log_file: Path, item_id: str, pdf_name: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a") as f:
        f.write(f"[{timestamp}] Downloaded: {item_id}/{pdf_name}\n")


def record_result(
    json_log: Path, item_id: str, status: str, filename: str = None, error: str = None
):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "item_id": item_id,
        "status": status,
        "filename": filename,
        "error": error,
    }
    existing = []
    if json_log.exists():
        try:
            with json_log.open("r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.append(record)
    with json_log.open("w") as f:
        json.dump(existing, f, indent=2)


def download_item(args):
    item_id, output_dir_str, log_file_str, json_log_str, include_derived = args
    output_dir = Path(output_dir_str)
    log_file = Path(log_file_str)
    json_log = Path(json_log_str)

    try:
        item = internetarchive.get_item(item_id)
        files = item.files
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False

        for file in files:
            name = file.get("name", "")
            if name.endswith(".pdf") and "wiki" not in name.lower():
                file_path = output_dir / name
                if not file_path.exists():
                    print(f"Downloading {item_id}/{name}")
                    success = item.download(
                        files=[name],
                        destdir=str(output_dir),
                        verbose=True,
                        retries=3,
                        ignore_existing=True,
                        no_directory=True,
                    )
                    if success:
                        create_log_entry(log_file, item_id, name)
                        record_result(json_log, item_id, "success", filename=name)
                        downloaded = True
                break

        if include_derived and not downloaded:
            print(f"Trying derived PDF for {item_id}")
            success = item.download(
                formats=["pdf"], destdir=str(output_dir), verbose=True, retries=2
            )
            if success:
                for file in output_dir.glob("*.pdf"):
                    if file.name.lower().endswith(".pdf"):
                        create_log_entry(log_file, item_id, file.name)
                        record_result(json_log, item_id, "success", filename=file.name)
                        downloaded = True
                        break

        if not downloaded:
            print(f"[Warning] No PDF available for {item_id}")
            record_result(json_log, item_id, "failed", error="No PDF found")

    except Exception as e:
        print(f"[ERROR] Failed to download {item_id}: {e}")
        record_result(json_log, item_id, "error", error=str(e))


def main():
    args = parse_args()
    resolved_language = resolve_language_name(args.language)
    print(f"Searching for language: {resolved_language} (from input: {args.language})")

    query = f'(language:{args.language} OR language:"{resolved_language}" OR subject:"{resolved_language}") AND (mediatype:texts)'

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "download.log"
    json_log = output_dir / "download_results.json"

    print("Querying archive.org...")
    results = internetarchive.search_items(query)

    download_jobs = [
        (
            result["identifier"],
            str(output_dir),
            str(log_file),
            str(json_log),
            args.include_derived,
        )
        for result in results
    ]

    print(f"Starting downloads of {len(download_jobs)} items...")

    with Pool(args.workers) as pool:
        pool.map(download_item, download_jobs)

    print(f"Finished. PDFs saved to: {output_dir}")
    print(f"Log written to: {log_file}")
    print(f"JSON summary written to: {json_log}")


if __name__ == "__main__":
    main()
