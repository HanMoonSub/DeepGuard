import cv2
import numpy as np


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def deprocess_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # (1, C, H, W -> H, W, C)
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    img = np.clip(img, 0, 1) # range(0~1)

    return np.float32(img)

def remove_padding_and_resize(cam, orig_shape, target_shape):
    orig_h, orig_w = orig_shape[:2] # (H,W,3)
    target_h, target_w = target_shape[2:] # (1,3,H,W)
    
    scale = max(target_h, target_w) / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Remove Padding
    unpadded_cam = cam[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
    return cv2.resize(unpadded_cam, (orig_w, orig_h))