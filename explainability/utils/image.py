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

def show_bbox_on_image(annotated_img: np.ndarray, cam_recovered: np.ndarray, binary_mask: np.ndarray, thickness: int = 1):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        roi_prob = float(np.max(cam_recovered[y:y + bh, x:x + bw]))

        r = int(min(255, 2 * roi_prob * 255))
        g = int(min(255, 2 * (1 - roi_prob) * 255))
        box_color = (r, g, 30)

        cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), box_color, thickness)

        draw_label(annotated_img, f"{roi_prob * 100:.0f}%", x, max(y - 5, 12), roi_prob)

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

def draw_label(img: np.ndarray, text: str, x: int, y: int, prob_ratio: float = 1.0):
    h, w = img.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.28, min(w,h) / 1000) # 이미지 크기 기반 동적 스케일
    thickness  = 1
    pad        = 4

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    rx1, ry1 = x - pad,      y - th - pad
    rx2, ry2 = x + tw + pad, y + baseline + pad

    r = int(min(255, 2 * prob_ratio * 255))
    g = int(min(255, 2 * (1 - prob_ratio) * 255))
    bg_color = (r, g, 30)  # RGB: green → red

    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
