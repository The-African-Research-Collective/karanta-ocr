"""
This script is used to convert PDF files to images.
"""

import os
import logging
import tempfile
from pathlib import Path
import argparse
from PIL import Image
from multiprocessing import Pool, cpu_count

from karanta.data.utils import convert_pdf2image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up logger
logger = logging.getLogger(__name__)


def process_single_pdf(pdf_path, output_base_path, output_format):
    """Process a single PDF file."""
    full_path, item = pdf_path
    combined_pages = []

    try:
        logger.info(f"Processing {item}")
        pdf_basename = os.path.splitext(item)[0]

        with tempfile.TemporaryDirectory() as path:
            pages = convert_pdf2image(data_path=full_path, output_dir=path)

            # Attach the PDF basename to each page
            for page in pages:
                combined_pages.append((page, pdf_basename))

            # Save pages
            for idx, (page, pdf_basename) in enumerate(combined_pages):
                filename = f"{pdf_basename}_pg_{idx}.{output_format.lower()}"
                save_path = output_base_path / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                page.save(save_path, output_format)

        return item, True
    except Exception as e:
        logger.error(f"Error processing {item}: {str(e)}")
        return item, False


def image_to_pdf(image_path, output_dir="output"):
    """
    Convert an image to PDF and save it to the specified output directory.
    The PDF will have the same name as the original image (without extension).

    Args:
        image_path (str): Path to the input image file
        output_dir (str): Directory to save the PDF (default: "output")

    Returns:
        str: Path to the created PDF file

    Raises:
        FileNotFoundError: If the input image doesn't exist
        ValueError: If the image format is not supported
    """

    # Check if input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    pdf_name = f"{base_name}.pdf"

    # Full output path
    output_path = os.path.join(output_dir, pdf_name)

    try:
        # Open and convert image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compatibility in PDF)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save as PDF
            img.save(output_path, "PDF", resolution=100.0)

        print(f"Successfully converted {image_path} to {output_path}")
        return output_path

    except Exception as e:
        raise ValueError(f"Error converting image to PDF: {str(e)}")


def main(args):
    output_base_path = Path(args.output_base_path)
    output_format = args.output_format.upper()

    # Collect all PDF files to process
    pdf_files = []
    if os.path.isfile(args.data_path):
        # If a single file is provided, add it to the list
        pdf_files.append((args.data_path, os.path.basename(args.data_path)))
    else:
        for item in os.listdir(args.data_path):
            full_path = os.path.join(args.data_path, item)
            if os.path.isfile(full_path):
                if item.lower().endswith(".pdf"):
                    pdf_files.append((full_path, item))
                elif item.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
                ):
                    # Convert image to PDF if needed
                    try:
                        image_to_pdf(full_path, output_base_path)
                        logger.info(f"Converted {item} to PDF.")
                    except Exception as e:
                        logger.error(f"Failed to convert {item} to PDF: {str(e)}")

    # Determine the number of processes to use
    num_processes = min(
        cpu_count(), len(pdf_files), args.num_processes
    )  # Cap at 8 processes

    if not pdf_files:
        logger.warning("No PDF files found in the specified directory")
        return

    logger.info(
        f"Processing {len(pdf_files)} PDF files using {num_processes} processes"
    )

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Process PDFs in parallel
        results = []
        for pdf_path in pdf_files:
            results.append(
                pool.apply_async(
                    process_single_pdf, args=(pdf_path, output_base_path, output_format)
                )
            )

        # Collect and log results
        completed = 0
        failed = 0
        for result in results:
            filename, success = result.get()  # Wait for result
            if success:
                completed += 1
            else:
                failed += 1

    logger.info(f"Processing complete: {completed} successful, {failed} failed")


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
        help="Image format for output files (default: JPEG). Supported formats: JPEG, PNG, BMP, GIF, TIFF, WEBP, PDF",
        choices=["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP", "PDF"],
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for conversion (default: 4).",
    )

    args = parser.parse_args()
    main(args)
