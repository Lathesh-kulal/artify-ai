import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def resize_image(image: Image.Image, max_dim: int = 768) -> Image.Image:
    """
    Resizes image maintaining aspect ratio such that the longest side does not exceed max_dim.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image

    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def preprocess_image(image: Image.Image, max_dim: int = 768, device: str = "cpu") -> torch.Tensor:
    """
    Prepares PIL Image for Fast Neural Style Transfer model.
    Values scaled to [0, 255] as expected by Johnson et al. models.
    """
    # Ensure RGB
    image = image.convert("RGB")
    image = resize_image(image, max_dim)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def preserve_original_colors(stylized_img: Image.Image, original_img: Image.Image) -> Image.Image:
    """
    Transfers original image colors to the stylized image using the YCrCb color space.
    The luminance (Y) comes from the stylized image, while chromatic channels (Cr, Cb)
    come from the original content image.
    """
    stylized_cv = cv2.cvtColor(np.array(stylized_img), cv2.COLOR_RGB2BGR)
    original_cv = cv2.cvtColor(np.array(original_img.resize(stylized_img.size, Image.Resampling.LANCZOS)), cv2.COLOR_RGB2BGR)

    stylized_ycrcb = cv2.cvtColor(stylized_cv, cv2.COLOR_BGR2YCrCb)
    original_ycrcb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2YCrCb)

    y_channel = stylized_ycrcb[:, :, 0]
    cr_channel = original_ycrcb[:, :, 1]
    cb_channel = original_ycrcb[:, :, 2]

    merged = cv2.merge([y_channel, cr_channel, cb_channel])
    result_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)


def blend_images(stylized_img: Image.Image, original_img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Blends the stylized image with the original image based on intensity (alpha).
    intensity = 1.0: 100% stylized
    intensity = 0.5: 50% stylized, 50% original
    """
    if intensity >= 1.0:
        return stylized_img
    
    matched_orig = original_img.resize(stylized_img.size, Image.Resampling.LANCZOS)
    return Image.blend(matched_orig, stylized_img, alpha=max(0.0, min(1.0, intensity)))


def postprocess_tensor(
    tensor: torch.Tensor,
    original_img: Image.Image = None,
    preserve_color: bool = False,
    intensity: float = 1.0
) -> Image.Image:
    """
    Converts model output tensor back into a clean PIL Image.
    Optionally preserves colors and adjusts style intensity.
    """
    output_np = tensor.squeeze(0).clamp(0, 255).cpu().detach().numpy()
    output_np = output_np.transpose(1, 2, 0).astype("uint8")
    result_image = Image.fromarray(output_np)

    if original_img is not None:
        if preserve_color:
            result_image = preserve_original_colors(result_image, original_img)
        if intensity < 1.0:
            result_image = blend_images(result_image, original_img, intensity)

    return result_image
