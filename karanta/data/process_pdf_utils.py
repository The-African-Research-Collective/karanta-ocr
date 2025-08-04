import base64
import subprocess
from typing import List


def get_pdf_media_box_width_height(
    local_pdf_path: str, page_num: int
) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A dictionary containing MediaBox dimensions or None if not found
    """
    # Construct the pdfinfo command to extract info for the specific page
    command = [
        "pdfinfo",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-box",
        "-enc",
        "UTF-8",
        local_pdf_path,
    ]

    # Run the command using subprocess
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_base64png(
    local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048
) -> str:
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(
                target_longest_image_dim * 72 / longest_dim
            ),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")
