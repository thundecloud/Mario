"""
Sprite replacement system - replaces Mario's head in all sprite frames.
Uses proportion-based head detection and full head erasure before compositing.
"""

import pygame as pg
import numpy as np
import cv2


def cv2_to_pygame(cv2_image_bgra):
    """Convert a BGRA OpenCV image (numpy array) to a Pygame surface."""
    rgba = cv2.cvtColor(cv2_image_bgra, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    surface = pg.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    return surface.convert_alpha()


def pygame_to_cv2(surface):
    """Convert a Pygame surface to a BGRA OpenCV image (numpy array)."""
    w, h = surface.get_size()
    raw = pg.image.tostring(surface, 'RGBA')
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bgra


def resize_face_to_head(face_bgra, head_width, head_height):
    """Resize face image to fit head region, maintaining aspect ratio and centering.
    Returns BGRA numpy array sized exactly (head_height, head_width, 4)."""
    if face_bgra is None:
        return None

    fh, fw = face_bgra.shape[:2]
    if fh == 0 or fw == 0:
        return None

    scale = min(head_width / fw, head_height / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))

    resized = cv2.resize(face_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in head region
    result = np.zeros((head_height, head_width, 4), dtype=np.uint8)
    x_off = (head_width - new_w) // 2
    y_off = (head_height - new_h) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return result


def alpha_blend(base, overlay, y_start, x_start):
    """Alpha-blend overlay onto base array in-place.
    Both are BGRA uint8 arrays."""
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]

    # Clip to bounds
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


def composite_face_on_sprite(sprite_surface, face_bgra, size_multiplier):
    """Composite face onto a single sprite frame.
    Automatically determines head region by proportion, erases original head,
    then blends face in.
    Returns new Pygame Surface."""
    if face_bgra is None:
        return sprite_surface

    w, h = sprite_surface.get_size()
    if w < 2 or h < 2:
        return sprite_surface  # Skip 0x0 placeholders

    # Determine head region by proportion
    original_h = round(h / size_multiplier)
    # Big mario (32px): head ~47%; Big crouch (22px): ~47%; Small mario (16px): ~56%
    if original_h > 20:
        head_ratio = 0.47
    else:
        head_ratio = 0.56
    head_h = max(1, int(h * head_ratio))

    # Convert sprite to numpy BGRA
    arr = pygame_to_cv2(sprite_surface)

    # Erase original head pixels (set to fully transparent)
    arr[0:head_h, :, :] = 0

    # Resize face to head region
    face_resized = resize_face_to_head(face_bgra, w, head_h)
    if face_resized is None:
        return cv2_to_pygame(arr)

    # Alpha blend face into head region
    alpha_blend(arr, face_resized, 0, 0)

    return cv2_to_pygame(arr)


class SpriteReplacer:
    """Manages replacing Mario's head across all sprite states and frames."""

    def __init__(self, size_multiplier=2.5):
        self.size_multiplier = size_multiplier
        self.face_bgra = None

    def set_face(self, face_bgra):
        """Set the face image to use for replacement.
        face_bgra: numpy array in BGRA format."""
        self.face_bgra = face_bgra

    def replace_head_in_frame(self, sprite_surface):
        """Replace head in a single sprite frame.
        Returns new Pygame Surface."""
        if self.face_bgra is None:
            return sprite_surface
        return composite_face_on_sprite(
            sprite_surface, self.face_bgra, self.size_multiplier
        )

    def replace_all_frames(self, mario_obj):
        """Replace heads in all of Mario's sprite frame lists.
        Modifies frames in-place."""
        if self.face_bgra is None:
            return

        # Process all small mario frames (right-facing)
        small_frame_lists = [
            mario_obj.right_small_normal_frames,
            mario_obj.right_small_green_frames,
            mario_obj.right_small_red_frames,
            mario_obj.right_small_black_frames,
        ]
        for frame_list in small_frame_lists:
            for i in range(len(frame_list)):
                frame_list[i] = self.replace_head_in_frame(frame_list[i])

        # Process all big mario frames (right-facing)
        big_frame_lists = [
            mario_obj.right_big_normal_frames,
            mario_obj.right_big_green_frames,
            mario_obj.right_big_red_frames,
            mario_obj.right_big_black_frames,
        ]
        for frame_list in big_frame_lists:
            for i in range(len(frame_list)):
                frame_list[i] = self.replace_head_in_frame(frame_list[i])

        # Process fire mario frames (right-facing)
        for i in range(len(mario_obj.right_fire_frames)):
            mario_obj.right_fire_frames[i] = self.replace_head_in_frame(
                mario_obj.right_fire_frames[i]
            )

        # Regenerate all left-facing frames by flipping
        self._regenerate_left_frames(mario_obj)

        # Update current frame references
        if mario_obj.big:
            if mario_obj.fire:
                mario_obj.right_frames = mario_obj.right_fire_frames
                mario_obj.left_frames = mario_obj.left_fire_frames
            else:
                mario_obj.right_frames = mario_obj.right_big_normal_frames
                mario_obj.left_frames = mario_obj.left_big_normal_frames
        else:
            mario_obj.right_frames = mario_obj.right_small_normal_frames
            mario_obj.left_frames = mario_obj.left_small_normal_frames

    def _regenerate_left_frames(self, mario_obj):
        """Regenerate all left-facing frames by flipping right frames."""
        frame_pairs = [
            (mario_obj.right_small_normal_frames, mario_obj.left_small_normal_frames),
            (mario_obj.right_small_green_frames, mario_obj.left_small_green_frames),
            (mario_obj.right_small_red_frames, mario_obj.left_small_red_frames),
            (mario_obj.right_small_black_frames, mario_obj.left_small_black_frames),
            (mario_obj.right_big_normal_frames, mario_obj.left_big_normal_frames),
            (mario_obj.right_big_green_frames, mario_obj.left_big_green_frames),
            (mario_obj.right_big_red_frames, mario_obj.left_big_red_frames),
            (mario_obj.right_big_black_frames, mario_obj.left_big_black_frames),
            (mario_obj.right_fire_frames, mario_obj.left_fire_frames),
        ]

        for right_frames, left_frames in frame_pairs:
            left_frames.clear()
            for frame in right_frames:
                flipped = pg.transform.flip(frame, True, False)
                left_frames.append(flipped)
