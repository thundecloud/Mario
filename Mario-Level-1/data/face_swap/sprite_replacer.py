"""
Sprite replacement system - replaces Mario's head in all sprite frames.
Analyzes the sprite sheet and composites user face onto Mario's body.
"""

import pygame as pg
import numpy as np
import cv2


# Head region definitions for each Mario sprite frame.
# Format: (y_offset_from_top, head_height) in original sprite pixels (before scaling).
# These define where the head starts and how tall it is within each sprite frame.

# Small Mario sprites are 12-16 x 16 pixels
# Head occupies roughly the top 8-9 pixels
SMALL_MARIO_HEAD = {
    'y_offset': 0,
    'head_height': 9,
    'sprite_height': 16,
}

# Big Mario sprites are 15-16 x 32 pixels
# Head occupies roughly the top 15 pixels
BIG_MARIO_HEAD = {
    'y_offset': 0,
    'head_height': 15,
    'sprite_height': 32,
}

# Big Mario crouching sprites are 16 x 22 pixels
# Head occupies roughly top 10 pixels
BIG_MARIO_CROUCH_HEAD = {
    'y_offset': 0,
    'head_height': 10,
    'sprite_height': 22,
}


def cv2_to_pygame(cv2_image_rgba):
    """Convert an RGBA OpenCV image (numpy array) to a Pygame surface."""
    # OpenCV uses BGRA, Pygame uses RGBA
    rgba = cv2.cvtColor(cv2_image_rgba, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    surface = pg.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    return surface.convert_alpha()


def pygame_to_cv2(surface):
    """Convert a Pygame surface to an RGBA OpenCV image (numpy array)."""
    w, h = surface.get_size()
    # Get raw pixel data as string
    raw = pg.image.tostring(surface, 'RGBA')
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    # Convert RGBA to BGRA for OpenCV
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bgra


def resize_face_to_head(face_rgba, head_width, head_height):
    """Resize face image to fit head region while maintaining aspect ratio.
    face_rgba: RGBA numpy array (OpenCV format, BGRA)
    Returns resized RGBA numpy array."""
    if face_rgba is None:
        return None

    fh, fw = face_rgba.shape[:2]
    # Calculate scale to fit within head region
    scale = min(head_width / fw, head_height / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)

    resized = cv2.resize(face_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in head region
    result = np.zeros((head_height, head_width, 4), dtype=np.uint8)
    x_offset = (head_width - new_w) // 2
    y_offset = (head_height - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result


def composite_face_on_sprite(sprite_surface, face_rgba, head_config, size_multiplier):
    """Composite processed face onto a single Mario sprite frame.
    sprite_surface: Pygame Surface of the Mario sprite
    face_rgba: RGBA numpy array (OpenCV BGRA format) of processed face
    head_config: dict with y_offset, head_height, sprite_height
    size_multiplier: the game's SIZE_MULTIPLIER value
    Returns new Pygame Surface with face composited."""
    if face_rgba is None:
        return sprite_surface

    w, h = sprite_surface.get_size()

    # Calculate head region in scaled sprite
    scaled_head_height = int(head_config['head_height'] * size_multiplier)
    scaled_y_offset = int(head_config['y_offset'] * size_multiplier)

    # Ensure head fits in sprite
    scaled_head_height = min(scaled_head_height, h - scaled_y_offset)

    if scaled_head_height <= 0 or w <= 0:
        return sprite_surface

    # Resize face to fit head region
    face_resized = resize_face_to_head(face_rgba, w, scaled_head_height)
    if face_resized is None:
        return sprite_surface

    # Convert face from OpenCV BGRA to RGBA for Pygame
    face_rgba_pg = cv2.cvtColor(face_resized, cv2.COLOR_BGRA2RGBA)
    face_surface = pg.image.frombuffer(
        face_rgba_pg.tobytes(), (face_rgba_pg.shape[1], face_rgba_pg.shape[0]), 'RGBA'
    ).convert_alpha()

    # Create new sprite surface
    new_sprite = sprite_surface.copy()
    new_sprite.blit(face_surface, (0, scaled_y_offset))

    return new_sprite


class SpriteReplacer:
    """Manages replacing Mario's head across all sprite states and frames."""

    def __init__(self, size_multiplier=2.5):
        self.size_multiplier = size_multiplier
        self.face_rgba = None  # Processed face in BGRA format

    def set_face(self, face_rgba):
        """Set the face image to use for replacement.
        face_rgba: RGBA numpy array in OpenCV BGRA format."""
        self.face_rgba = face_rgba

    def replace_head_in_frame(self, sprite_surface, is_big=False, is_crouching=False):
        """Replace the head in a single sprite frame.
        sprite_surface: Pygame Surface
        is_big: True for big/fire mario
        is_crouching: True for crouching frame
        Returns new Pygame Surface."""
        if self.face_rgba is None:
            return sprite_surface

        if is_crouching:
            head_config = BIG_MARIO_CROUCH_HEAD
        elif is_big:
            head_config = BIG_MARIO_HEAD
        else:
            head_config = SMALL_MARIO_HEAD

        return composite_face_on_sprite(
            sprite_surface, self.face_rgba, head_config, self.size_multiplier
        )

    def replace_all_frames(self, mario_obj):
        """Replace heads in all of Mario's sprite frame lists.
        mario_obj: the Mario sprite instance.
        Modifies frames in-place."""
        if self.face_rgba is None:
            return

        # Process small mario frames (right-facing)
        small_frame_lists = [
            mario_obj.right_small_normal_frames,
            mario_obj.right_small_green_frames,
            mario_obj.right_small_red_frames,
            mario_obj.right_small_black_frames,
        ]

        for frame_list in small_frame_lists:
            for i in range(len(frame_list)):
                frame_list[i] = self.replace_head_in_frame(
                    frame_list[i], is_big=False
                )

        # Process big mario frames (right-facing)
        big_frame_lists = [
            mario_obj.right_big_normal_frames,
            mario_obj.right_big_green_frames,
            mario_obj.right_big_red_frames,
            mario_obj.right_big_black_frames,
        ]

        for frame_list in big_frame_lists:
            for i in range(len(frame_list)):
                is_crouch = (i == 7 and len(frame_list) > 7)
                frame_list[i] = self.replace_head_in_frame(
                    frame_list[i], is_big=True, is_crouching=is_crouch
                )

        # Process fire mario frames (right-facing)
        for i in range(len(mario_obj.right_fire_frames)):
            is_crouch = (i == 7)
            mario_obj.right_fire_frames[i] = self.replace_head_in_frame(
                mario_obj.right_fire_frames[i], is_big=True, is_crouching=is_crouch
            )

        # Regenerate all left-facing frames by flipping right frames
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
