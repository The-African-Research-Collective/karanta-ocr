import os
import tempfile
import random
from pathlib import Path
import argparse

from newspaper_parser.utils import split_pdf2image_and_add_to_dataframe, check_for_text_in_image, check_if_image_requires_segmentation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_base_path", "--output_path", required=True)
    parser.add_argument("--train_test_split", type=float, default=0.0)
    return parser.parse_args()

def main():
    args = parse_args()
    output_base_path = Path(args.output_base_path)
    combined_pages = []

    for item in os.listdir(args.data_path):
        full_path = os.path.join(args.data_path, item)
        if os.path.isfile(full_path) and item.lower().endswith('.pdf'):
            pdf_basename = os.path.splitext(item)[0]

            with tempfile.TemporaryDirectory() as path:
                pages = split_pdf2image_and_add_to_dataframe(data_path=args.data_path, output_dir=path)
                # Attach the PDF basename to each page
                for page in pages:
                    combined_pages.append((page, pdf_basename))

    segment_pages = []
    no_segment_pages = []

    for page, pdf_basename in combined_pages:
        if not check_for_text_in_image(image=page):
            continue

        if check_if_image_requires_segmentation(image=page):
            segment_pages.append((page, pdf_basename))
        else:
            no_segment_pages.append((page, pdf_basename))

    random.shuffle(segment_pages)
    random.shuffle(no_segment_pages)

    segment_test_count = int(len(segment_pages) * args.train_test_split)
    no_segment_test_count = int(len(no_segment_pages) * args.train_test_split)

    segment_test = segment_pages[:segment_test_count]
    segment_train = segment_pages[segment_test_count:]

    no_segment_test = no_segment_pages[:no_segment_test_count]
    no_segment_train = no_segment_pages[no_segment_test_count:]

    all_data = []
    for page, basename in segment_train:
        all_data.append((page, basename, "segment", "train"))
    for page, basename in segment_test:
        all_data.append((page, basename, "segment", "test"))
    for page, basename in no_segment_train:
        all_data.append((page, basename, "no_segment", "train"))
    for page, basename in no_segment_test:
        all_data.append((page, basename, "no_segment", "test"))

    # Save pages
    for idx, (page, pdf_basename, label, split) in enumerate(all_data):
        filename = f"{pdf_basename}_pg_{idx}.jpg"
        save_path = output_base_path / split / label / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        page.save(save_path, 'JPEG')


if __name__ == "__main__":
    main()