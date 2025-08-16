import logging
import base64

from dataclasses import dataclass
from abc import ABC

from PIL import Image
from io import BytesIO

from karanta.training.utils import SingleDatapoint
from karanta.data.process_pdf_utils import render_pdf_to_base64png

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BasePipelineStep(ABC):
    """This is the base class for all the pipeline steps."""

    def __call__(self, *args, **kwargs):
        """Call the step with the given arguments."""
        pass


@dataclass(frozen=True, slots=True)
class PDF2ImageStep(BasePipelineStep):
    """Pipeline step that renders PDF to image."""

    target_longest_image_dim: int

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """Render PDF to image."""
        print(sample)
        # # Render PDF to image
        base64_png = render_pdf_to_base64png(
            str(sample.pdf_path),
            page_num=1,
            target_longest_image_dim=self.target_longest_image_dim,
        )
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))

        # # Update sample
        sample.image = image

        return sample
