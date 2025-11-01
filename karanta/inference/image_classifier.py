import torch
import numpy as np
from PIL import Image
from typing import Union
from transformers import pipeline

# Load the model only once
_classifier_pipeline = pipeline(
    "image-classification",
    model="taresco/newspaper_classifier_segformer",
    device=0 if torch.cuda.is_available() else -1
)

def load_image(image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Convert various image input types to a PIL RGB image."""
    if isinstance(image, str):
        # image path or URL
        if image.startswith("http://") or image.startswith("https://"):
            from urllib.request import urlopen
            return Image.open(urlopen(image)).convert("RGB")
        return Image.open(image).convert("RGB")
    
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    
    elif isinstance(image, torch.Tensor):
        # Convert tensor to numpy first
        np_image = image.detach().cpu().numpy()
        if np_image.ndim == 3 and np_image.shape[0] in (1, 3):  # C x H x W
            np_image = np.transpose(np_image, (1, 2, 0))
        return Image.fromarray((np_image * 255).astype(np.uint8)).convert("RGB")
    
    else:
        raise ValueError("Unsupported image type")

def predict_layout(image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> dict:
    """
    Predict if the image is a 'segment' or 'no_segment'.
    
    Returns:
        dict: {'label': 'segment' | 'no_segment', 'score': float}
    """
    pil_image = load_image(image)
    predictions = _classifier_pipeline(pil_image)
    if not predictions:
        raise ValueError("No predictions returned by the classifier pipeline.")
    best = max(predictions, key=lambda x: x['score'])
    return best


