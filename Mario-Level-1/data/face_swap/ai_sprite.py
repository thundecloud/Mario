"""
AI Sprite Generation - converts face photos to pixel art sprites via OpenAI GPT Image API.
Falls back to local pixelation when API is unavailable.
"""

import cv2
import numpy as np
import base64
import os
import logging

logger = logging.getLogger(__name__)


def _load_api_key():
    """Load OpenAI API key from environment or .env file."""
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        return key

    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        # Search upward for .env
        for parent in ['.', '..', '../..']:
            env_path = os.path.join(os.path.dirname(__file__), parent, '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
                break
        key = os.environ.get('OPENAI_API_KEY')
    except ImportError:
        pass

    return key


def bgra_to_png_bytes(face_bgra):
    """Convert BGRA numpy array to PNG bytes."""
    success, png_data = cv2.imencode('.png', face_bgra)
    if not success:
        return None
    return png_data.tobytes()


def decode_base64_to_bgra(b64_string):
    """Decode base64 PNG to BGRA numpy array."""
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Ensure BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def generate_sprite_face(face_bgra):
    """Generate pixel art sprite face via OpenAI GPT Image API.
    face_bgra: BGRA numpy array of the face.
    Returns BGRA numpy array on success, None on failure."""
    api_key = _load_api_key()
    if not api_key:
        logger.info("No OPENAI_API_KEY found, falling back to local pixelation")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, falling back to local pixelation")
        return None

    png_bytes = bgra_to_png_bytes(face_bgra)
    if png_bytes is None:
        return None

    try:
        client = OpenAI(api_key=api_key)

        result = client.images.edit(
            model="gpt-image-1",
            image=png_bytes,
            prompt=(
                "Convert this face photo into a 16-bit retro pixel art game character "
                "head sprite. Keep the person's key facial features recognizable. "
                "Use a limited color palette similar to NES/SNES games. The result "
                "should look like a classic Mario-style character head with clear "
                "pixel grid. Keep transparent background."
            ),
            size="1024x1024",
            quality="medium",
        )

        # Extract base64 image data
        if result.data and len(result.data) > 0:
            b64 = result.data[0].b64_json
            if b64:
                return decode_base64_to_bgra(b64)

            # If URL-based response
            url = getattr(result.data[0], 'url', None)
            if url:
                return _download_image_url(url)

    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")

    return None


def _download_image_url(url):
    """Download image from URL and convert to BGRA."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as resp:
            img_bytes = resp.read()
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    except Exception as e:
        logger.warning(f"Failed to download image: {e}")
        return None


def local_pixel_fallback(face_bgra, target_size=(16, 16)):
    """Local fallback: simple pixelation via downscale + nearest-neighbor upscale.
    Also applies NES-like color reduction and saturation boost."""
    if face_bgra is None:
        return None

    h, w = face_bgra.shape[:2]

    bgr = face_bgra[:, :, :3]
    alpha = face_bgra[:, :, 3]

    # Downscale
    small = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    small_alpha = cv2.resize(alpha, target_size, interpolation=cv2.INTER_AREA)

    # Color reduction (k-means)
    small = _reduce_colors(small, n_colors=16)

    # Saturation boost
    small = _boost_saturation(small, factor=1.3)

    # Upscale with nearest neighbor
    pixel_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    pixel_alpha = cv2.resize(small_alpha, (w, h), interpolation=cv2.INTER_NEAREST)

    return np.dstack([pixel_bgr, pixel_alpha])


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
