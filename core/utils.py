import os
import io
from PIL import Image

def get_image_bytes(image: Image.Image, format: str = "JPEG", quality: int = 95) -> bytes:
    """
    Encodes a PIL Image into bytes for downloading or streaming.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def save_image(image: Image.Image, output_path: str, format: str = "JPEG", quality: int = 95) -> str:
    """
    Saves a PIL Image to disk, creating destination folders if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path, format=format, quality=quality)
    return output_path
