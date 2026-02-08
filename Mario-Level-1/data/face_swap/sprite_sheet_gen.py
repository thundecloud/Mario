"""
Sprite Sheet Generator - AI generates NORMAL form pixel heads,
then palette mapping auto-generates all color variant rows.

Flow:
  User photo -> AI generates 2 pixel heads (small + big Mario, 4x resolution)
  -> Color align to original NES palette
  -> Per-frame composite (new head + original body) -> write NORMAL rows
  -> Extract color mapping from original sheet -> auto-generate Green/Red/Black/Fire variant rows
  -> Output complete modified sprite sheet (Pygame Surface)
  -> mario.py uses new sheet directly, get_image() coordinates unchanged
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Frame coordinates in the sprite sheet (from mario.py get_image calls) ──

# NORMAL small Mario (y=32, height 16px) - 11 frames
SMALL_NORMAL = [
    (178, 32, 12, 16),  # [0] standing
    (80,  32, 15, 16),  # [1] walk1
    (96,  32, 16, 16),  # [2] walk2
    (112, 32, 16, 16),  # [3] walk3
    (144, 32, 16, 16),  # [4] jump
    (130, 32, 14, 16),  # [5] skid
    (160, 32, 15, 16),  # [6] death
    (320, 8,  16, 24),  # [7] grow transition (special height!)
    (241, 33, 16, 16),  # [8] shrink transition
    (194, 32, 12, 16),  # [9] flag1
    (210, 33, 12, 16),  # [10] flag2
]

# NORMAL big Mario (y=0, height 32px) - 11 frames
BIG_NORMAL = [
    (176, 0,  16, 32),  # [0] standing
    (81,  0,  16, 32),  # [1] walk1
    (97,  0,  15, 32),  # [2] walk2
    (113, 0,  15, 32),  # [3] walk3
    (144, 0,  16, 32),  # [4] jump
    (128, 0,  16, 32),  # [5] skid
    (336, 0,  16, 32),  # [6] throw
    (160, 10, 16, 22),  # [7] crouch (special y and height!)
    (272, 2,  16, 29),  # [8] transition
    (193, 2,  16, 30),  # [9] flag1
    (209, 2,  16, 29),  # [10] flag2
]

# Color variant Y offsets (X coordinates same as NORMAL)
SMALL_VARIANTS = {'green': 224, 'red': 272, 'black': 176}  # first 6 frames only
BIG_VARIANTS = {'green': 192, 'red': 240, 'black': 144}    # first 8 frames only
FIRE_Y = 48  # big Mario fire, full 11 frames

# NORMAL row base Y for offset calculation
SMALL_NORMAL_BASE_Y = 32
BIG_NORMAL_BASE_Y = 0


def pygame_to_numpy(surface):
    """Convert Pygame Surface to BGRA numpy array."""
    import pygame as pg
    w, h = surface.get_size()
    raw = pg.image.tostring(surface, 'RGBA')
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bgra


def numpy_to_pygame(bgra_arr):
    """Convert BGRA numpy array to Pygame Surface."""
    import pygame as pg
    rgba = cv2.cvtColor(bgra_arr, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    surface = pg.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    return surface.convert_alpha()


def _resize_and_center(head_img, target_w, target_h):
    """Resize head image to fit target dimensions, maintaining aspect ratio, centered."""
    if head_img is None:
        return None
    fh, fw = head_img.shape[:2]
    if fh == 0 or fw == 0:
        return None

    scale = min(target_w / fw, target_h / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))

    resized = cv2.resize(head_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    result = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result


def _alpha_blend(base, overlay, y_start, x_start):
    """Alpha-blend overlay onto base array in-place. Both are BGRA uint8."""
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]

    y_end = min(y_start + oh, bh)
    x_end = min(x_start + ow, bw)
    oy_end = y_end - y_start
    ox_end = x_end - x_start

    if oy_end <= 0 or ox_end <= 0:
        return

    roi = base[y_start:y_end, x_start:x_end]
    over = overlay[:oy_end, :ox_end]

    alpha = over[:, :, 3:4].astype(np.float32) / 255.0
    inv_alpha = 1.0 - alpha

    roi[:, :, :3] = (over[:, :, :3].astype(np.float32) * alpha +
                     roi[:, :, :3].astype(np.float32) * inv_alpha).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], over[:, :, 3])


class PaletteMapper:
    """Extracts color mappings between NORMAL and variant rows from the original sprite sheet,
    then applies those mappings to AI-generated frames."""

    def __init__(self, sheet_arr):
        self.sheet = sheet_arr
        self._mappings = {}

    def get_mapping(self, dst_variant, category):
        """Extract normal->variant color mapping from original sheet.
        Compares the standing frame (index 0) pixel by pixel."""
        key = (dst_variant, category)
        if key in self._mappings:
            return self._mappings[key]

        if category == 'small':
            src_y = SMALL_NORMAL_BASE_Y
            if dst_variant == 'fire':
                self._mappings[key] = {}
                return {}
            dst_y = SMALL_VARIANTS[dst_variant]
            frame_x, _, w, h = SMALL_NORMAL[0]
        else:
            src_y = BIG_NORMAL_BASE_Y
            if dst_variant == 'fire':
                dst_y = FIRE_Y
            else:
                dst_y = BIG_VARIANTS[dst_variant]
            frame_x, _, w, h = BIG_NORMAL[0]

        color_map = {}
        src_frame = self.sheet[src_y:src_y + h, frame_x:frame_x + w]
        dst_frame = self.sheet[dst_y:dst_y + h, frame_x:frame_x + w]

        for y in range(h):
            for x in range(w):
                if src_frame[y, x, 3] > 128 and dst_frame[y, x, 3] > 128:
                    sc = tuple(src_frame[y, x, :3].tolist())
                    dc = tuple(dst_frame[y, x, :3].tolist())
                    if sc != (0, 0, 0):  # skip pure black outline
                        color_map[sc] = dc

        self._mappings[key] = color_map
        return color_map

    def apply_mapping(self, frame_bgra, color_map, tolerance=30):
        """Apply color mapping to a frame using nearest-neighbor matching."""
        if not color_map:
            return frame_bgra.copy()

        result = frame_bgra.copy()
        src_colors = np.array(list(color_map.keys()), dtype=np.float32)
        dst_colors = np.array(list(color_map.values()), dtype=np.uint8)

        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if result[y, x, 3] < 128:
                    continue
                pixel = result[y, x, :3].astype(np.float32)
                dists = np.sqrt(np.sum((src_colors - pixel) ** 2, axis=1))
                idx = np.argmin(dists)
                if dists[idx] <= tolerance:
                    result[y, x, :3] = dst_colors[idx]
        return result


class SpriteSheetGenerator:
    """AI generates NORMAL form + palette mapping generates variants -> complete sprite sheet."""

    def generate(self, face_bgra, original_sheet_surface, progress_cb=None):
        """Main entry point.
        face_bgra: BGRA numpy array of the user's face.
        original_sheet_surface: Pygame Surface of mario_bros.png sprite sheet.
        Returns modified Pygame Surface on success, None on failure."""
        sheet_arr = pygame_to_numpy(original_sheet_surface)

        # Save original sheet for palette extraction before we modify it
        original_arr = sheet_arr.copy()

        # Phase 1: AI generate pixel heads (2 API calls)
        if progress_cb:
            progress_cb('Generating pixel heads...')
        heads = self._generate_heads(face_bgra)
        if heads is None:
            logger.info("AI generation failed, trying local fallback")
            heads = self._local_fallback_heads(face_bgra)
        if heads is None:
            logger.warning("All head generation methods failed")
            return None

        # Phase 2: Color align to NES palette
        if progress_cb:
            progress_cb('Aligning colors...')
        heads['small'] = self._align_to_palette(heads['small'], original_arr, 'small')
        heads['big'] = self._align_to_palette(heads['big'], original_arr, 'big')

        # Phase 3: Composite NORMAL rows - new head + original body
        if progress_cb:
            progress_cb('Compositing frames...')
        self._composite_normal_row(sheet_arr, heads, SMALL_NORMAL, 'small')
        self._composite_normal_row(sheet_arr, heads, BIG_NORMAL, 'big')

        # Phase 4: Palette map to generate variant rows
        if progress_cb:
            progress_cb('Generating color variants...')
        palette_mapper = PaletteMapper(original_arr)  # Use ORIGINAL sheet for mapping
        self._map_variants(sheet_arr, palette_mapper, 'small')
        self._map_variants(sheet_arr, palette_mapper, 'big')

        return numpy_to_pygame(sheet_arr)

    def _generate_heads(self, face_bgra):
        """Call OpenAI gpt-image-1 to generate 2 pixel heads."""
        from .ai_sprite import _load_api_key
        api_key = _load_api_key()
        if not api_key:
            logger.info("No OPENAI_API_KEY found")
            return None

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed")
            return None

        client = OpenAI(api_key=api_key)
        heads = {}

        # Small Mario: original head ~12x9px -> 4x = 48x36px generation
        heads['small'] = self._call_ai(client, face_bgra,
            target_4x=(48, 36), final=(12, 9),
            prompt=(
                "Convert this face into a pixel art character head sprite in the exact "
                "style of NES Super Mario Bros. The head should be front-facing, using "
                "classic Mario color scheme: brown hair, skin tone face, red cap on top. "
                "Use a limited palette of 5-6 colors with clear pixel grid. "
                "Transparent background. The result will be downscaled to 12x9 pixels, "
                "so keep features chunky (at least 4x4 pixels per detail)."
            ))

        # Big Mario: original head ~16x15px -> 4x = 64x60px generation
        heads['big'] = self._call_ai(client, face_bgra,
            target_4x=(64, 60), final=(16, 15),
            prompt=(
                "Convert this face into a pixel art character head sprite in the exact "
                "style of NES Super Mario Bros. The head should be front-facing, using "
                "classic Mario color scheme: brown hair, skin tone face, red cap on top. "
                "Use a limited palette of 5-6 colors with clear pixel grid. "
                "Transparent background. The result will be downscaled to 16x15 pixels, "
                "so keep features chunky (at least 4x4 pixels per detail)."
            ))

        if all(v is not None for v in heads.values()):
            return heads
        return None

    def _call_ai(self, client, face_bgra, target_4x, final, prompt):
        """Single AI call: face -> pixel head."""
        try:
            from .ai_sprite import bgra_to_png_bytes, decode_base64_to_bgra

            # Crop face core area, scale to target_4x
            face_resized = cv2.resize(face_bgra, target_4x, interpolation=cv2.INTER_AREA)
            png_bytes = bgra_to_png_bytes(face_resized)
            if png_bytes is None:
                return None

            result = client.images.edit(
                model="gpt-image-1",
                image=png_bytes,
                prompt=prompt,
                size="1024x1024",
                quality="medium",
            )

            if result.data and len(result.data) > 0:
                b64 = result.data[0].b64_json
                if b64:
                    img = decode_base64_to_bgra(b64)
                    if img is not None:
                        return self._postprocess_ai_result(img, final)

                url = getattr(result.data[0], 'url', None)
                if url:
                    from .ai_sprite import _download_image_url
                    img = _download_image_url(url)
                    if img is not None:
                        return self._postprocess_ai_result(img, final)

        except Exception as e:
            logger.warning(f"AI head generation failed: {e}")
        return None

    def _postprocess_ai_result(self, img, final_size):
        """Post-process AI result: auto-crop, downscale, quantize."""
        # Auto-crop non-transparent area
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            rows = np.any(alpha > 128, axis=1)
            cols = np.any(alpha > 128, axis=0)
            if rows.any() and cols.any():
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                img = img[ymin:ymax + 1, xmin:xmax + 1]

        # Downscale to final size
        img = cv2.resize(img, final_size, interpolation=cv2.INTER_AREA)

        # Color quantize to 6 colors
        bgr = img[:, :, :3]
        from .ai_sprite import _reduce_colors
        bgr = _reduce_colors(bgr, n_colors=6)
        img[:, :, :3] = bgr

        return img

    def _local_fallback_heads(self, face_bgra):
        """Local fallback: downscale + quantize."""
        from .ai_sprite import _reduce_colors, _boost_saturation
        heads = {}
        for name, size in [('small', (12, 9)), ('big', (16, 15))]:
            bgr = face_bgra[:, :, :3]
            alpha = face_bgra[:, :, 3]
            small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            small_a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
            small = _reduce_colors(small, n_colors=6)
            small = _boost_saturation(small, factor=1.3)
            heads[name] = np.dstack([small, small_a])
        return heads

    def _align_to_palette(self, head_img, sheet_arr, category):
        """Align AI head colors to the original NES palette extracted from the sheet."""
        if head_img is None:
            return head_img

        # Extract palette colors from the NORMAL standing frame
        if category == 'small':
            fx, fy, fw, fh = SMALL_NORMAL[0]
        else:
            fx, fy, fw, fh = BIG_NORMAL[0]

        ref_frame = sheet_arr[fy:fy + fh, fx:fx + fw]

        # Collect unique non-transparent, non-black colors
        palette = set()
        for y in range(ref_frame.shape[0]):
            for x in range(ref_frame.shape[1]):
                if ref_frame[y, x, 3] > 128:
                    c = tuple(ref_frame[y, x, :3].tolist())
                    if c != (0, 0, 0):
                        palette.add(c)

        if not palette:
            return head_img

        palette_arr = np.array(list(palette), dtype=np.float32)
        result = head_img.copy()

        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if result[y, x, 3] < 128:
                    continue
                pixel = result[y, x, :3].astype(np.float32)
                if tuple(result[y, x, :3].tolist()) == (0, 0, 0):
                    continue  # keep black outline
                dists = np.sqrt(np.sum((palette_arr - pixel) ** 2, axis=1))
                idx = np.argmin(dists)
                result[y, x, :3] = palette_arr[idx].astype(np.uint8)

        return result

    def _composite_normal_row(self, sheet, heads, frame_list, category):
        """Write AI heads into NORMAL row for each frame."""
        head_img = heads[category]

        for (fx, fy, fw, fh) in frame_list:
            # Calculate head height for this frame
            if fh <= 16:
                head_h = max(1, int(fh * 0.56))   # small Mario
            elif fh <= 24:
                head_h = max(1, int(fh * 0.50))   # transition frame
            else:
                head_h = max(1, int(fh * 0.47))   # big Mario

            # Resize head to frame width x head_h, maintaining aspect ratio
            head_resized = _resize_and_center(head_img, fw, head_h)
            if head_resized is None:
                continue

            # Clear original head region
            sheet[fy:fy + head_h, fx:fx + fw, :] = 0

            # Write new head
            _alpha_blend(sheet, head_resized, fy, fx)

    def _map_variants(self, sheet, mapper, category):
        """Apply palette mapping from modified NORMAL frames to generate variant rows."""
        if category == 'small':
            normal_frames = SMALL_NORMAL
            variants = SMALL_VARIANTS  # green/red/black, first 6 frames only
            variant_counts = {'green': 6, 'red': 6, 'black': 6}
        else:
            normal_frames = BIG_NORMAL
            variants = dict(BIG_VARIANTS)
            variants['fire'] = FIRE_Y
            variant_counts = {'green': 8, 'red': 8, 'black': 8, 'fire': 11}

        for variant_name, variant_base_y in variants.items():
            color_map = mapper.get_mapping(variant_name, category)
            max_idx = variant_counts[variant_name]

            for i in range(min(max_idx, len(normal_frames))):
                fx, fy, fw, fh = normal_frames[i]

                # Read modified NORMAL frame
                normal_frame = sheet[fy:fy + fh, fx:fx + fw].copy()

                # Apply color mapping
                variant_frame = mapper.apply_mapping(normal_frame, color_map)

                # Calculate variant Y coordinate
                # The variant rows use the same X coords but different Y base
                if category == 'small':
                    # Normal base Y is 32; variant rows offset the same way
                    vy = variant_base_y + (fy - SMALL_NORMAL_BASE_Y)
                else:
                    vy = variant_base_y + (fy - BIG_NORMAL_BASE_Y)

                # Write variant frame, clamp to sheet bounds
                if vy >= 0 and vy + fh <= sheet.shape[0]:
                    sheet[vy:vy + fh, fx:fx + fw] = variant_frame
