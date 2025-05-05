"""
This script is used to convert PDF files to images.
"""

import os
import logging
import tempfile
from pathlib import Path
import argparse

from karanta.data.utils import split_pdf2image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(args: argparse.ArgumentParser):
    output_base_path = Path(args.output_base_path)
    output_format = args.output_format.upper()

    for item in os.listdir(args.data_path):
        combined_pages = []

        logger.info(f"Processing {item}")
        full_path = os.path.join(args.data_path, item)
        if os.path.isfile(full_path) and item.lower().endswith(".pdf"):
            pdf_basename = os.path.splitext(item)[0]

            with tempfile.TemporaryDirectory() as path:
                pages = split_pdf2image(data_path=full_path, output_dir=path)
                # Attach the PDF basename to each page
                for page in pages:
                    combined_pages.append((page, pdf_basename))

        # Save pages
        for idx, (page, pdf_basename) in enumerate(combined_pages):
            filename = f"{pdf_basename}_pg_{idx}.{output_format.lower()}"
            save_path = output_base_path / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            page.save(save_path, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF files to images.")
    parser.add_argument(
        "--data_path", required=True, help="Path to the directory containing PDF files."
    )
    parser.add_argument(
        "--output_base_path",
        "--output_path",
        required=True,
        help="Path to the directory to save output images.",
    )
    parser.add_argument(
        "--output_format",
        default="JPEG",
        help="Image format for output files (default: JPEG). Supported formats: JPEG, PNG, BMP, GIF, TIFF, WEBP.",
        choices=["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"],
    )

    args = parser.parse_args()
    main(args)
