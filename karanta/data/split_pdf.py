import os
import logging
import argparse

from pathlib import Path
from pypdf import PdfReader, PdfWriter
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Set up logger
logger = logging.getLogger(__name__)


def process_single_pdf(pdf_path, output_base_path):
    """Process a single PDF file."""
    full_path, item = pdf_path

    try:
        logger.info(f"Processing {item}")
        with open(full_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            page = 0
            total_pages = len(reader.pages)

            logger.info(f"Processing {item} with {total_pages} pages")

            while page < total_pages:
                writer = PdfWriter()
                writer.add_page(reader.pages[page])

                # Save the current page to a new PDF file
                output_pdf_path = output_base_path / f"{item}_page_{page}.pdf"
                with open(output_pdf_path, "wb") as output_pdf_file:
                    writer.write(output_pdf_file)

                page += 1

        return item, True
    except Exception as e:
        logger.error(f"Error processing {item}: {str(e)}")
        return item, False


def main(args):
    output_base_path = Path(args.output_base_path)

    # Collect all PDF files to process
    pdf_files = []
    if os.path.isfile(args.data_path):
        # If a single file is provided, add it to the list
        pdf_files.append((args.data_path, os.path.basename(args.data_path)))
    else:
        for item in os.listdir(args.data_path):
            full_path = os.path.join(args.data_path, item)
            if os.path.isfile(full_path) and item.lower().endswith(".pdf"):
                pdf_files.append((full_path, item))

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
                pool.apply_async(process_single_pdf, args=(pdf_path, output_base_path))
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
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for conversion (default: 4).",
    )

    args = parser.parse_args()
    main(args)
