import os
import re
import logging
import json
import numpy as np
import argparse
from matplotlib.path import Path
import multiprocessing
from tqdm import tqdm
from functools import partial
import cv2
from datasets import Dataset, DatasetDict, Image as HFImage

# Set up logger with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_annotations(json_file_path):
    """Load annotations from a JSON file."""
    logger.info(f"Loading annotations from {json_file_path}")
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        logger.info(
            f"Successfully loaded annotations with {len(data.get('images', []))} images and {len(data.get('annotations', []))} annotations"
        )
        return data
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        raise


def find_image_by_name(annotations_data, image_name):
    """Find image data in annotations by filename."""
    simplified_name = simplify_filename(image_name)
    for image in annotations_data["images"]:
        annotation_name = simplify_filename(
            image.get("extra", {}).get("name", "")
        ) or simplify_filename(image.get("file_name", ""))
        if annotation_name == simplified_name:
            return image

    logger.warning(f"Image '{image_name}' not found in annotations")
    return None


def find_annotations_by_image_id(annotations_data, image_id):
    """Find annotations for a specific image ID."""
    annotations = [
        ann for ann in annotations_data["annotations"] if ann["image_id"] == image_id
    ]
    logger.debug(f"Found {len(annotations)} annotations for image ID {image_id}")
    return annotations


def create_segmentation_bitmap(
    image_path, annotations_data, image_name, output_path=None, class_colors=None
):
    """Create a visualization segmentation bitmap for an image based on its annotations."""
    logger.debug(f"Creating segmentation bitmap for {image_name}")

    # Find image data
    image_data = find_image_by_name(annotations_data, image_name)
    if image_data is None:
        logger.error(f"Image '{image_name}' not found in annotations")
        raise ValueError(f"Image '{image_name}' not found in annotations")

    height, width = image_data["height"], image_data["width"]
    annotations = find_annotations_by_image_id(annotations_data, image_data["id"])
    logger.debug(f"Processing {len(annotations)} annotations for image {image_name}")

    # Create empty segmentation bitmap (initialized to 0 for background)
    segmentation = np.zeros((height, width), dtype=np.uint8)

    # Fill the segmentation bitmap based on polygon segmentations
    for ann in annotations:
        class_id = ann["category_id"]

        if (
            "segmentation" in ann
            and ann["segmentation"]
            and isinstance(ann["segmentation"][0], list)
        ):
            # Process polygon segmentation
            polygon = np.array(ann["segmentation"][0]).reshape(-1, 2)
            y, x = np.mgrid[:height, :width]
            points = np.vstack((x.flatten(), y.flatten())).T
            mask = Path(polygon).contains_points(points).reshape(height, width)
            segmentation[mask] = class_id
        elif "bbox" in ann:
            # Process bounding box
            x, y, w, h = map(int, ann["bbox"])
            x, y = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            segmentation[y:y2, x:x2] = class_id

    # Initialize or update class colors dictionary
    if class_colors is None:
        class_colors = {}

    # Process unique classes
    for class_id in np.unique(segmentation):
        if class_id not in class_colors:
            if class_id == 0:  # Background
                class_colors[class_id] = [0, 0, 0]  # Black
            else:
                # Generate a new distinct color using HSV
                hue = (len(class_colors) * 137.5) % 360  # Golden ratio * 360
                h, s, v = hue / 60, 0.75, 0.9
                i, f = int(h), h - int(h)
                p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))

                rgb = [
                    [v, t, p],
                    [q, v, p],
                    [p, v, t],
                    [p, q, v],
                    [t, p, v],
                    [v, p, q],
                ][i % 6]

                class_colors[class_id] = [int(c * 255) for c in rgb]

    # Create visualization with color lookup
    color_lut = np.zeros((max(np.unique(segmentation)) + 1, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        if class_id < len(color_lut):
            color_lut[class_id] = color

    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            visualization[y, x] = color_lut[segmentation[y, x]]

    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, visualization)
        logger.debug(f"Saved segmentation bitmap to {output_path}")

    return visualization, class_colors


def process_single_image(
    image_file, images_dir, annotations_data, output_dir, class_colors
):
    """Process a single image and create visualization segmentation bitmap."""
    try:
        image_path = os.path.join(images_dir, image_file)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")

        # Create segmentation bitmap
        _, updated_class_colors = create_segmentation_bitmap(
            image_path, annotations_data, base_name, output_path, class_colors.copy()
        )

        logger.info(f"Processed {image_file}")
        return image_file, updated_class_colors
    except Exception as e:
        logger.error(f"Error processing {image_file}: {e}")
        return image_file, class_colors


def batch_process_images(images_dir, json_file_path, output_dir, num_processes=None):
    """Process all images in parallel using multiprocessing."""
    # Setup processing environment
    num_processes = num_processes or multiprocessing.cpu_count()
    logger.info(f"Starting batch processing with {num_processes} processes")

    annotations_data = load_annotations(json_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg"))]
    total_images = len(image_files)
    logger.info(f"Found {total_images} images to process")

    # Setup shared dictionary for class colors
    manager = multiprocessing.Manager()
    shared_class_colors = manager.dict()

    # Create process function with fixed arguments
    process_func = partial(
        process_single_image,
        images_dir=images_dir,
        annotations_data=annotations_data,
        output_dir=output_dir,
        class_colors=shared_class_colors,
    )

    # Process images in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, image_files),
                total=total_images,
                desc="Processing images",
            )
        )

    # Merge class colors from all processes
    final_class_colors = {}
    for _, colors in results:
        final_class_colors.update(
            {k: v for k, v in colors.items() if k not in final_class_colors}
        )

    logger.info(f"Batch processing completed. Processed {total_images} images.")
    return final_class_colors


def simplify_filename(filename):
    # Match the base pattern like "newspaper-12_pg_0" before the complex identifiers
    match = re.match(r"(newspaper-\d+_pg_\d+)", filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0]  # Fallback to just removing extension


def create_dataset(image_paths, label_paths):
    """Helper function to create dataset from image and label paths."""
    logger.info(f"Creating dataset with {len(image_paths)} samples")
    dataset = Dataset.from_dict(
        {"image": sorted(image_paths), "label": sorted(label_paths)}
    )
    dataset = dataset.cast_column("image", HFImage())
    dataset = dataset.cast_column("label", HFImage())
    return dataset


def main(args):
    logger.info("Starting image processing and dataset creation")

    # Process images with multiprocessing
    batch_process_images(
        args.image_dir, args.annotations_json, args.output_dir, args.num_processes
    )

    # Get processed files
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith((".jpg"))]
    label_files = [f for f in os.listdir(args.output_dir) if f.endswith(".png")]
    logger.info(
        f"Found {len(image_files)} image files and {len(label_files)} label files"
    )

    # Create mapping and matching pairs
    label_map = {simplify_filename(f): f for f in label_files}
    image_paths_train = []
    label_paths_train = []

    for img_file in image_files:
        simple_name = simplify_filename(img_file)
        if simple_name in label_map:
            image_paths_train.append(os.path.join(args.image_dir, img_file))
            label_paths_train.append(
                os.path.join(args.output_dir, label_map[simple_name])
            )

    logger.info(
        f"Created dataset with {len(image_paths_train)} matching image-label pairs"
    )

    # Create and push dataset
    train_dataset = create_dataset(image_paths_train, label_paths_train)
    dataset = DatasetDict({"train": train_dataset})

    logger.info(f"Pushing dataset to Hugging Face Hub with ID: {args.hub_dataset_id}")
    dataset.push_to_hub(args.hub_dataset_id)
    logger.info(
        f"Dataset successfully pushed to Hugging Face Hub: {args.hub_dataset_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert annotations to segmentation bitmaps and create dataset."
    )
    parser.add_argument(
        "--annotations_json",
        required=True,
        type=str,
        help="Path to the annotations JSON file.",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        type=str,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default="output/dataset_labels/",
        type=str,
        help="Path to the directory to save segmentation bitmaps.",
    )
    parser.add_argument(
        "--hub_dataset_id",
        required=True,
        type=str,
        help="Dataset ID for the Hugging Face Hub.",
    )
    parser.add_argument(
        "--num_processes",
        required=False,
        type=int,
        default=4,
        help="Number of processes to use for multiprocessing. Default is 4",
    )

    args = parser.parse_args()
    main(args)
