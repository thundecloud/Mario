"""
Style transfer engine - provides pixel, original, and cartoon style processing.
"""

import cv2
import numpy as np


def apply_pixel_style(face_rgba, target_size=(16, 16)):
    """Apply pixel art style to face image.
    Downscale then upscale with nearest neighbor interpolation.
    face_rgba: RGBA numpy array
    target_size: intermediate low-res size for pixelation
    Returns RGBA numpy array at original size."""
    if face_rgba is None:
        return None

    h, w = face_rgba.shape[:2]

    # Split into color and alpha
    bgr = face_rgba[:, :, :3]
    alpha = face_rgba[:, :, 3]

    # Downscale to pixel art size
    small = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    small_alpha = cv2.resize(alpha, target_size, interpolation=cv2.INTER_AREA)

    # Optional: apply NES-like color palette reduction
    small = _reduce_colors(small, n_colors=16)

    # Increase saturation and contrast for NES look
    small = _boost_saturation(small, factor=1.3)

    # Upscale back with nearest neighbor (creates pixel blocks)
    pixel_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    pixel_alpha = cv2.resize(small_alpha, (w, h), interpolation=cv2.INTER_NEAREST)

    # Combine back to RGBA
    result = np.dstack([pixel_bgr, pixel_alpha])
    return result


def apply_original_style(face_rgba):
    """Apply minimal processing to keep original photo look.
    Smooth edges slightly and adjust for game context.
    face_rgba: RGBA numpy array
    Returns RGBA numpy array."""
    if face_rgba is None:
        return None

    bgr = face_rgba[:, :, :3].copy()
    alpha = face_rgba[:, :, 3].copy()

    # Slight edge smoothing
    bgr = cv2.bilateralFilter(bgr, 5, 50, 50)

    # Slight brightness/contrast adjustment
    bgr = cv2.convertScaleAbs(bgr, alpha=1.1, beta=10)

    # Smooth alpha edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 2)

    result = np.dstack([bgr, alpha])
    return result


def apply_cartoon_style(face_rgba):
    """Apply cartoon/cel-shaded style to face image.
    Uses bilateral filter for smoothing + edge detection.
    face_rgba: RGBA numpy array
    Returns RGBA numpy array."""
    if face_rgba is None:
        return None

    bgr = face_rgba[:, :, :3].copy()
    alpha = face_rgba[:, :, 3].copy()

    # Step 1: Apply bilateral filter multiple times for smooth cartoon look
    for _ in range(3):
        bgr = cv2.bilateralFilter(bgr, 9, 75, 75)

    # Step 2: Color quantization - reduce number of colors
    bgr = _reduce_colors(bgr, n_colors=8)

    # Step 3: Edge detection using adaptive threshold
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )

    # Step 4: Combine edges with color image
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(bgr, edges_bgr)

    # Boost saturation for vibrant cartoon look
    cartoon = _boost_saturation(cartoon, factor=1.4)

    result = np.dstack([cartoon, alpha])
    return result


def _reduce_colors(bgr_image, n_colors=16):
    """Reduce number of colors using k-means clustering."""
    data = bgr_image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(bgr_image.shape)


def _boost_saturation(bgr_image, factor=1.3):
    """Boost color saturation of BGR image."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Style name to function mapping
STYLE_FUNCTIONS = {
    'pixel': apply_pixel_style,
    'original': apply_original_style,
    'cartoon': apply_cartoon_style,
}


def apply_style(face_rgba, style_name='pixel'):
    """Apply named style to face image.
    style_name: 'pixel', 'original', or 'cartoon'
    Returns RGBA numpy array."""
    func = STYLE_FUNCTIONS.get(style_name)
    if func is None:
        raise ValueError(f"Unknown style: {style_name}. Available: {list(STYLE_FUNCTIONS.keys())}")
    return func(face_rgba)
