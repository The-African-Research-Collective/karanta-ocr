"""
This code is an adaptation of https://github.com/allenai/olmocr/blob/main/olmocr/prompts/anchor.py
"""

import re
import ftfy  # package used to fix text encoding issues
import random
from dataclasses import dataclass
from typing import List, Literal

from pypdf import PdfReader
from pypdf.generic import RectangleObject


def _transform_point(x, y, m):
    x_new = m[0] * x + m[2] * y + m[4]
    y_new = m[1] * x + m[3] * y + m[5]
    return x_new, y_new


def _mult(m: List[float], n: List[float]) -> List[float]:
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


@dataclass(frozen=True)
class Element:
    pass


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_rectangle(rect: RectangleObject) -> "BoundingBox":
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])


@dataclass(frozen=True)
class TextElement(Element):
    text: str
    x: float
    y: float


@dataclass(frozen=True)
class ImageElement(Element):
    name: str
    bbox: BoundingBox


@dataclass(frozen=True)
class PageReport:
    mediabox: BoundingBox
    text_elements: List[TextElement]
    image_elements: List[ImageElement]


def _pdf_report(local_pdf_path: str, page_num: int = 1) -> PageReport:
    """
    This function is used to extract text and image elements from a PDF page.
    It uses the pyPDF library to read the PDF and extract the necessary information.
    It doesnt work too well for PDFs that aren't image based but works fine for image based PDFs.
    It extracts Text elements containing text and their coordinates, and Image elements containing
    the image name and bounding box coordinates.

    e.g. PageReport(mediabox=BoundingBox(x0=0.0, y0=0.0, x1=841.68, y1=1190.4),
    text_elements=[TextElement(text='', x=0.0, y=0.0), TextElement(text='', x=0.0, y=0.0)],
    image_elements=[ImageElement(name='/Im0', bbox=BoundingBox(x0=0.0, y0=0.0, x1=841.68, y1=1190.4))])
    """
    reader = PdfReader(local_pdf_path)
    page = reader.pages[page_num - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    text_elements, image_elements = [], []

    def visitor_body(text, cm, tm, font_dict, font_size):
        """
        function to be called when extracting some text at some position.
        It has five arguments: text, current transformation matrix,
        text matrix, font-dictionary and font-size.
        The font-dictionary may be None in case of unknown fonts.
        If not None it may e.g. contain key "/BaseFont" with value "/Arial,Bold".
        Ignored with a warning in "layout" mode.
        """
        txt2user = _mult(tm, cm)
        text_elements.append(TextElement(text, txt2user[4], txt2user[5]))

    def visitor_op(op, args, cm, tm):
        """
        function to be called before processing an operation.
        It has four arguments: operator, operand-arguments,
        current transformation matrix and text matrix.
        """
        if op == b"Do":
            xobject_name = args[0]
            xobject = xobjects.get(xobject_name)
            if xobject and xobject["/Subtype"] == "/Image":
                # Compute image bbox
                # The image is placed according to the CTM
                _width = xobject.get("/Width")
                _height = xobject.get("/Height")
                x0, y0 = _transform_point(0, 0, cm)
                x1, y1 = _transform_point(1, 1, cm)
                image_elements.append(
                    ImageElement(
                        xobject_name,
                        BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                    )
                )

    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=image_elements,
    )


def _merge_image_elements(
    images: List[ImageElement], tolerance: float = 0.5
) -> List[ImageElement]:
    n = len(images)
    parent = list(range(n))  # Initialize Union-Find parent pointers

    def find(i):
        # Find with path compression
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent_i = parent[i]
            parent[i] = root
            i = parent_i
        return root

    def union(i, j):
        # Union by attaching root of one tree to another
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def bboxes_overlap(b1: BoundingBox, b2: BoundingBox, tolerance: float) -> bool:
        # Compute horizontal and vertical distances between boxes
        h_dist = max(0, max(b1.x0, b2.x0) - min(b1.x1, b2.x1))
        v_dist = max(0, max(b1.y0, b2.y0) - min(b1.y1, b2.y1))
        # Check if distances are within tolerance
        return h_dist <= tolerance and v_dist <= tolerance

    # Union overlapping images
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i].bbox, images[j].bbox, tolerance):
                union(i, j)

    # Group images by their root parent
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Merge images in the same group
    merged_images = []
    for indices in groups.values():
        # Initialize merged bounding box
        merged_bbox = images[indices[0]].bbox
        merged_name = images[indices[0]].name

        for idx in indices[1:]:
            bbox = images[idx].bbox
            # Expand merged_bbox to include the current bbox
            merged_bbox = BoundingBox(
                x0=min(merged_bbox.x0, bbox.x0),
                y0=min(merged_bbox.y0, bbox.y0),
                x1=max(merged_bbox.x1, bbox.x1),
                y1=max(merged_bbox.y1, bbox.y1),
            )
            # Optionally, update the name
            merged_name += f"+{images[idx].name}"

        merged_images.append(ImageElement(name=merged_name, bbox=merged_bbox))

    # Return the merged images along with other elements
    return merged_images


def _cap_split_string(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    head_length = max_length // 2 - 3
    tail_length = head_length

    head = text[:head_length].rsplit(" ", 1)[0] or text[:head_length]
    tail = text[-tail_length:].split(" ", 1)[-1] or text[-tail_length:]

    return f"{head} ... {tail}"


def _cleanup_element_text(element_text: str) -> str:
    MAX_TEXT_ELEMENT_LENGTH = 250
    TEXT_REPLACEMENTS = {"[": "\\[", "]": "\\]", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    text_replacement_pattern = re.compile(
        "|".join(re.escape(key) for key in TEXT_REPLACEMENTS.keys())
    )

    element_text = ftfy.fix_text(element_text).strip()

    # Replace square brackets with escaped brackets and other escaped chars
    element_text = text_replacement_pattern.sub(
        lambda match: TEXT_REPLACEMENTS[match.group(0)], element_text
    )

    return _cap_split_string(element_text, MAX_TEXT_ELEMENT_LENGTH)


def _linearize_pdf_report(report: PageReport, max_length: int = 4000) -> str:
    result = ""
    result += f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"

    if max_length < 20:
        return result

    images = _merge_image_elements(report.image_elements)

    # Process image elements
    image_strings = []
    for element in images:
        image_str = f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]\n"
        # Use element's unique identifier (e.g., id or position) for comparison
        image_strings.append((element, image_str))

    # Process text elements
    text_strings = []
    for element in report.text_elements:  # type: ignore
        if len(element.text.strip()) == 0:  # type: ignore
            continue

        element_text = _cleanup_element_text(element.text)  # type: ignore

        text_str = f"[{element.x:.0f}x{element.y:.0f}]{element_text}\n"  # type: ignore
        text_strings.append((element, text_str))

    # Combine all elements with their positions for sorting
    all_elements: list[tuple[str, ImageElement, str, tuple[float, float]]] = []
    for elem, s in image_strings:
        position = (elem.bbox.x0, elem.bbox.y0)
        all_elements.append(("image", elem, s, position))
    for elem, s in text_strings:
        position = (elem.x, elem.y)  # type: ignore
        all_elements.append(("text", elem, s, position))

    # Calculate total length
    total_length = len(result) + sum(len(s) for _, _, s, _ in all_elements)

    if total_length <= max_length:
        # Include all elements
        all_elements.sort(key=lambda x: (x[3][1]), reverse=True)  # Sort by y-coordinate

        for _, _, s, _ in all_elements:
            result += s
        return result

    # Identify elements with min/max coordinates
    edge_elements = set()

    if images:
        min_x0_image = min(images, key=lambda e: e.bbox.x0)
        max_x1_image = max(images, key=lambda e: e.bbox.x1)
        min_y0_image = min(images, key=lambda e: e.bbox.y0)
        max_y1_image = max(images, key=lambda e: e.bbox.y1)
        edge_elements.update([min_x0_image, max_x1_image, min_y0_image, max_y1_image])

    if report.text_elements:
        text_elements = [e for e in report.text_elements if len(e.text.strip()) > 0]
        if text_elements:
            min_x_text = min(text_elements, key=lambda e: e.x)
            max_x_text = max(text_elements, key=lambda e: e.x)
            min_y_text = min(text_elements, key=lambda e: e.y)
            max_y_text = max(text_elements, key=lambda e: e.y)
            edge_elements.update([min_x_text, max_x_text, min_y_text, max_y_text])  # type: ignore

    # Keep track of element IDs to prevent duplication
    selected_element_ids = set()
    selected_elements = []

    # Include edge elements first
    for elem_type, elem, s, position in all_elements:
        if elem in edge_elements and id(elem) not in selected_element_ids:
            selected_elements.append((elem_type, elem, s, position))
            selected_element_ids.add(id(elem))

    # Calculate remaining length
    current_length = len(result) + sum(len(s) for _, _, s, _ in selected_elements)
    _remaining_length = max_length - current_length

    # Exclude edge elements from the pool
    remaining_elements = [
        (elem_type, elem, s, position)
        for elem_type, elem, s, position in all_elements
        if id(elem) not in selected_element_ids
    ]

    # Sort remaining elements by their positions (e.g., x-coordinate and then y-coordinate)
    # remaining_elements.sort(key=lambda x: (x[3][0], x[3][1]))

    # Shuffle remaining elements randomly
    random.shuffle(remaining_elements)

    # Add elements until reaching max_length
    for elem_type, elem, s, position in remaining_elements:
        if current_length + len(s) > max_length:
            break
        selected_elements.append((elem_type, elem, s, position))
        selected_element_ids.add(id(elem))
        current_length += len(s)

    # Sort selected elements by their positions to maintain logical order
    selected_elements.sort(key=lambda x: (x[3][0], x[3][1]))

    # Build the final result
    for _, _, s, _ in selected_elements:
        result += s

    return result


def get_anchor_text(
    local_pdf_path: str,
    page: int,
    pdf_engine: Literal["pdfreport"],
    target_length: int = 4000,
) -> str:
    if pdf_engine == "pdfreport":
        return _linearize_pdf_report(
            _pdf_report(local_pdf_path, page), max_length=target_length
        )
    else:
        raise ValueError(f"Unsupported PDF engine: {pdf_engine}")


if __name__ == "__main__":
    local_pdf_path = "/Users/odunayoogundepo/Downloads/Agbeyewo.pdf"
    page_num = 2

    print(get_anchor_text(local_pdf_path, page_num, pdf_engine="pdfreport"))
