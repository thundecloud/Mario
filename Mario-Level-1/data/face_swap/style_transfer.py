"""
Style transfer engine - provides sprite art and original style processing.
Sprite art uses AI generation (OpenAI GPT Image) with local fallback.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_sprite_style(face_bgra):
    """Apply pixel art sprite style to face image.
    Tries AI generation first, falls back to local pixelation.
    face_bgra: BGRA numpy array
    Returns BGRA numpy array."""
    if face_bgra is None:
        return None

    # Try AI generation
    from .ai_sprite import generate_sprite_face, local_pixel_fallback

    result = generate_sprite_face(face_bgra)
    if result is not None:
        logger.info("AI sprite generation succeeded")
        return result

    # Fallback to local pixelation
    logger.info("Using local pixel fallback")
    return local_pixel_fallback(face_bgra)


def apply_original_style(face_bgra):
    """Apply minimal processing to keep original photo look.
    face_bgra: BGRA numpy array
    Returns BGRA numpy array."""
    if face_bgra is None:
        return None

    bgr = face_bgra[:, :, :3].copy()
    alpha = face_bgra[:, :, 3].copy()

    # Slight edge smoothing
    bgr = cv2.bilateralFilter(bgr, 5, 50, 50)

    # Slight brightness/contrast adjustment
    bgr = cv2.convertScaleAbs(bgr, alpha=1.1, beta=10)

    # Smooth alpha edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 2)

    return np.dstack([bgr, alpha])


# Style name to function mapping
STYLE_FUNCTIONS = {
    'sprite': apply_sprite_style,
    'original': apply_original_style,
}


def apply_style(face_bgra, style_name='sprite'):
    """Apply named style to face image.
    style_name: 'sprite' or 'original'
    Returns BGRA numpy array."""
    func = STYLE_FUNCTIONS.get(style_name)
    if func is None:
        raise ValueError(f"Unknown style: {style_name}. Available: {list(STYLE_FUNCTIONS.keys())}")
    return func(face_bgra)
