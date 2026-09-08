import time
import torch
from PIL import Image
from typing import Tuple, Dict, Any

from core.model_loader import load_style_model
from core.image_processor import preprocess_image, postprocess_tensor

# In-memory model cache for fast re-runs
_MODEL_CACHE: Dict[str, Any] = {}

def get_cached_model(style_name: str, device: str):
    cache_key = f"{style_name}_{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_style_model(style_name, device=device)
    return _MODEL_CACHE[cache_key]


def apply_style_transfer(
    content_image: Image.Image,
    style_name: str,
    device: str = "cpu",
    max_dim: int = 768,
    preserve_color: bool = False,
    style_intensity: float = 1.0
) -> Tuple[Image.Image, float]:
    """
    Applies neural style transfer on content_image using the specified style model.
    Returns:
        (stylized_image, inference_duration_seconds)
    """
    start_time = time.time()

    # Load model
    model = get_cached_model(style_name, device)

    # Preprocess image
    input_tensor = preprocess_image(content_image, max_dim=max_dim, device=device)

    # Inference (no grad)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess
    result_image = postprocess_tensor(
        output_tensor,
        original_img=content_image,
        preserve_color=preserve_color,
        intensity=style_intensity
    )

    elapsed_time = time.time() - start_time
    return result_image, elapsed_time
