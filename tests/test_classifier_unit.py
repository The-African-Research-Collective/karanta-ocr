import pytest
from karanta.inference.image_classifier import predict_layout
from PIL import Image
import numpy as np
import torch
import os

test_image_path = "tests/sample.jpg"  # place a test image in this path

@pytest.mark.parametrize("input_type", ["path", "pil", "np", "tensor"])
def test_predict_layout(input_type):
    assert os.path.exists(test_image_path), f"Test image not found: {test_image_path}"
    if input_type == "path":
        result = predict_layout(test_image_path)
    elif input_type == "pil":
        result = predict_layout(Image.open(test_image_path))
    elif input_type == "np":
        img = np.array(Image.open(test_image_path))
        result = predict_layout(img)
    elif input_type == "tensor":
        img = torch.tensor(np.array(Image.open(test_image_path)) / 255.0).permute(2, 0, 1).float()
        result = predict_layout(img)

    assert "label" in result and "score" in result
    assert result["label"] in ["segment", "no_segment"]
    assert 0 <= result["score"] <= 1.0
