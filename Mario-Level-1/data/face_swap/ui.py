"""
Face Swap UI - Pygame-based interface for face capture and style selection.
Integrates face capture, detection, style transfer, and sprite replacement.
"""

import pygame as pg
import cv2
import numpy as np
import math
import sys
import os
import threading

# Add parent paths so we can import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .face_capture import CameraCapture, upload_photo
from .face_detector import FaceDetector
from .style_transfer import apply_style, STYLE_FUNCTIONS
from .sprite_replacer import SpriteReplacer


class FaceSwapUI:
    """Pygame-based UI for face swap setup before game starts."""

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (70, 130, 200)
    DARK_BLUE = (40, 80, 150)
    LIGHT_BLUE = (100, 170, 240)
    GREEN = (70, 200, 100)
    DARK_GREEN = (40, 150, 60)
    RED = (200, 70, 70)
    DARK_RED = (150, 40, 40)
    GRAY = (180, 180, 180)
    DARK_GRAY = (100, 100, 100)
    BG_COLOR = (40, 44, 52)
    CARD_COLOR = (55, 60, 70)
    GOLD = (255, 215, 0)

    def __init__(self, screen):
        self.screen = screen
        self.screen_w, self.screen_h = screen.get_size()
        self.clock = pg.time.Clock()
        self.detector = FaceDetector()
        self.replacer = SpriteReplacer()

        self.face_image = None      # Raw BGR image from capture
        self.face_rgba = None       # Extracted face RGBA (BGRA format)
        self.styled_face = None     # After style applied (BGRA format)
        self.selected_style = 'sprite'
        self.state = 'main_menu'    # main_menu, preview, style_select, manual_select

        # Style cache: {style_name: bgra_array}
        self._style_cache = {}

        # Loading state for async style generation
        self._loading = False
        self._loading_thread = None
        self._loading_result = None
        self._loading_style = None
        self._loading_dots = 0
        self._loading_timer = 0

        # Manual selection state
        self._select_start = None
        self._select_end = None
        self._selecting = False
        self._manual_photo_surface = None
        self._manual_photo_rect = None
        self._manual_scale = 1.0

        # Font setup
        self.title_font = pg.font.SysFont('Microsoft YaHei', 48, bold=True)
        self.button_font = pg.font.SysFont('Microsoft YaHei', 28)
        self.label_font = pg.font.SysFont('Microsoft YaHei', 22)
        self.small_font = pg.font.SysFont('Microsoft YaHei', 18)

    def run(self):
        """Main UI loop. Returns (styled_face_rgba, style_name) or (None, None) if cancelled."""
        running = True

        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None, None
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        if self.state == 'main_menu':
                            return None, None
                        elif self.state == 'manual_select':
                            self.state = 'preview' if self.face_rgba else 'main_menu'
                        else:
                            self.state = 'main_menu'
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.state == 'manual_select':
                            self._handle_manual_select_events(event)
                        elif not self._loading:
                            result = self._handle_click(event.pos)
                            if result is not None:
                                return result
                elif event.type == pg.MOUSEMOTION:
                    if self.state == 'manual_select' and self._selecting:
                        self._select_end = event.pos
                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button == 1 and self.state == 'manual_select' and self._selecting:
                        self._selecting = False
                        self._select_end = event.pos

            # Check if async loading finished
            self._check_loading_done()

            self.screen.fill(self.BG_COLOR)
            if self.state == 'main_menu':
                self._draw_main_menu()
            elif self.state == 'preview':
                self._draw_preview()
            elif self.state == 'style_select':
                self._draw_style_select()
            elif self.state == 'manual_select':
                self._draw_manual_select()

            pg.display.flip()
            self.clock.tick(30)

        return None, None

    def _check_loading_done(self):
        """Check if async style generation thread has completed."""
        if self._loading and self._loading_thread is not None:
            if not self._loading_thread.is_alive():
                result = self._loading_result
                style = self._loading_style
                self._loading = False
                self._loading_thread = None
                self._loading_result = None

                if result is not None and style is not None:
                    self._style_cache[style] = result
                    if self.selected_style == style:
                        self.styled_face = result

    def _handle_click(self, pos):
        """Handle mouse click at position. Returns result tuple or None."""
        x, y = pos

        if self.state == 'main_menu':
            # Camera button
            btn_y = 260
            if self._is_in_button(x, y, self.screen_w//2, btn_y, 280, 60):
                self._do_camera_capture()
                return None
            # Upload button
            btn_y = 340
            if self._is_in_button(x, y, self.screen_w//2, btn_y, 280, 60):
                self._do_upload()
                return None
            # Skip button (play without face swap)
            btn_y = 440
            if self._is_in_button(x, y, self.screen_w//2, btn_y, 280, 60):
                return (None, None)

        elif self.state == 'preview':
            # Confirm button
            btn_y = 500
            if self._is_in_button(x, y, self.screen_w//2 + 150, btn_y, 140, 50):
                self.state = 'style_select'
                self._apply_selected_style()
                return None
            # Retake button
            if self._is_in_button(x, y, self.screen_w//2 - 150, btn_y, 140, 50):
                self.state = 'main_menu'
                self.face_image = None
                self.face_rgba = None
                self._style_cache.clear()
                return None
            # Manual Select button
            if self._is_in_button(x, y, self.screen_w//2, btn_y, 150, 50):
                self._enter_manual_select()
                return None

        elif self.state == 'style_select':
            # Style cards - 2 styles centered with 220px spacing
            styles = [('sprite', 'Sprite Art'), ('original', 'Original')]
            start_x = self.screen_w // 2 - 185  # Center 2 cards
            for i, (style_key, _) in enumerate(styles):
                sx = start_x + i * 220
                sy = 160
                if sx <= x <= sx + 150 and sy <= y <= sy + 230:
                    self.selected_style = style_key
                    self._apply_selected_style()
                    return None

            # Confirm play button
            btn_y = 500
            if self._is_in_button(x, y, self.screen_w//2, btn_y, 280, 60):
                if self.styled_face is not None:
                    self.detector.close()
                    return (self.styled_face, self.selected_style)

            # Back button
            btn_y = 500
            if self._is_in_button(x, y, self.screen_w//2 - 200, btn_y, 120, 50):
                self.state = 'preview'
                return None

        return None

    def _is_in_button(self, mx, my, cx, cy, w, h):
        """Check if (mx, my) is inside a button centered at (cx, cy) with size (w, h)."""
        return (cx - w//2 <= mx <= cx + w//2) and (cy - h//2 <= my <= cy + h//2)

    def _do_camera_capture(self):
        """Initiate camera capture using Pygame-embedded camera view."""
        cam = CameraCapture(self.screen, self.detector)
        image, bbox = cam.run()

        if image is not None:
            self.face_image = image
            self.face_rgba = self.detector.extract_face(image, bbox=bbox)
            if self.face_rgba is not None:
                self.state = 'preview'
                self._style_cache.clear()
            else:
                # Auto-detection failed, enter manual select mode
                self._enter_manual_select()

    def _do_upload(self):
        """Initiate photo upload."""
        image = upload_photo()
        if image is not None:
            self.face_image = image
            self.face_rgba = self.detector.extract_face(image)
            if self.face_rgba is not None:
                self.state = 'preview'
                self._style_cache.clear()
            else:
                # Auto-detection failed, enter manual select mode
                self._enter_manual_select()

    def _apply_selected_style(self):
        """Apply currently selected style to face, using cache if available."""
        if self.face_rgba is None:
            return

        style = self.selected_style

        # Check cache
        if style in self._style_cache:
            self.styled_face = self._style_cache[style]
            return

        # For sprite style, run async since AI call may be slow
        if style == 'sprite' and not self._loading:
            self._loading = True
            self._loading_style = style
            self._loading_result = None
            self._loading_timer = pg.time.get_ticks()
            self._loading_dots = 0

            face_copy = self.face_rgba.copy()
            def _generate():
                self._loading_result = apply_style(face_copy, style)
            self._loading_thread = threading.Thread(target=_generate, daemon=True)
            self._loading_thread.start()
        else:
            # Original style is fast, run sync
            result = apply_style(self.face_rgba, style)
            self._style_cache[style] = result
            self.styled_face = result

    def _cv2_to_pg_surface(self, cv2_img, max_size=200):
        """Convert OpenCV image to Pygame surface, scaled to max_size."""
        if cv2_img is None:
            return None

        if len(cv2_img.shape) == 2:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

        if cv2_img.shape[2] == 4:
            # BGRA -> RGBA
            rgba = cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA)
        else:
            # BGR -> RGB, add alpha
            rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])

        h, w = rgba.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

        surface = pg.image.frombuffer(rgba.tobytes(), (new_w, new_h), 'RGBA')
        return surface.convert_alpha()

    def _draw_button(self, text, cx, cy, w, h, color, hover_color=None, text_color=None):
        """Draw a rounded button centered at (cx, cy)."""
        if text_color is None:
            text_color = self.WHITE

        mx, my = pg.mouse.get_pos()
        is_hover = self._is_in_button(mx, my, cx, cy, w, h)
        btn_color = hover_color if (is_hover and hover_color) else color

        rect = pg.Rect(cx - w//2, cy - h//2, w, h)
        pg.draw.rect(self.screen, btn_color, rect, border_radius=10)

        # Border
        border_color = tuple(min(c + 30, 255) for c in btn_color)
        pg.draw.rect(self.screen, border_color, rect, 2, border_radius=10)

        text_surf = self.button_font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=(cx, cy))
        self.screen.blit(text_surf, text_rect)

    def _draw_main_menu(self):
        """Draw the main menu screen."""
        # Title
        title = self.title_font.render("Mario Face Swap", True, self.GOLD)
        title_rect = title.get_rect(center=(self.screen_w // 2, 80))
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.label_font.render("Replace Mario's head with your face!", True, self.GRAY)
        sub_rect = subtitle.get_rect(center=(self.screen_w // 2, 140))
        self.screen.blit(subtitle, sub_rect)

        # Instructions
        instr = self.small_font.render("Choose how to capture your face:", True, self.LIGHT_BLUE)
        instr_rect = instr.get_rect(center=(self.screen_w // 2, 200))
        self.screen.blit(instr, instr_rect)

        # Buttons
        self._draw_button("Camera Capture", self.screen_w//2, 260, 280, 60,
                          self.BLUE, self.LIGHT_BLUE)
        self._draw_button("Upload Photo", self.screen_w//2, 340, 280, 60,
                          self.GREEN, self.DARK_GREEN)
        self._draw_button("Skip (No Face Swap)", self.screen_w//2, 440, 280, 60,
                          self.DARK_GRAY, self.GRAY)

        # Footer
        footer = self.small_font.render("Press ESC to quit", True, self.DARK_GRAY)
        footer_rect = footer.get_rect(center=(self.screen_w // 2, self.screen_h - 30))
        self.screen.blit(footer, footer_rect)

    def _draw_preview(self):
        """Draw the face preview screen."""
        title = self.title_font.render("Face Detected!", True, self.GREEN)
        title_rect = title.get_rect(center=(self.screen_w // 2, 60))
        self.screen.blit(title, title_rect)

        # Draw original photo
        if self.face_image is not None:
            orig_surf = self._cv2_to_pg_surface(self.face_image, 250)
            if orig_surf:
                label = self.label_font.render("Original Photo", True, self.GRAY)
                label_rect = label.get_rect(center=(self.screen_w // 4, 120))
                self.screen.blit(label, label_rect)

                orig_rect = orig_surf.get_rect(center=(self.screen_w // 4, 300))
                # Background card
                card = pg.Rect(orig_rect.x - 10, orig_rect.y - 10,
                               orig_rect.width + 20, orig_rect.height + 20)
                pg.draw.rect(self.screen, self.CARD_COLOR, card, border_radius=8)
                self.screen.blit(orig_surf, orig_rect)

        # Draw extracted face
        if self.face_rgba is not None:
            face_surf = self._cv2_to_pg_surface(self.face_rgba, 250)
            if face_surf:
                label = self.label_font.render("Extracted Face", True, self.GRAY)
                label_rect = label.get_rect(center=(3 * self.screen_w // 4, 120))
                self.screen.blit(label, label_rect)

                face_rect = face_surf.get_rect(center=(3 * self.screen_w // 4, 300))
                card = pg.Rect(face_rect.x - 10, face_rect.y - 10,
                               face_rect.width + 20, face_rect.height + 20)
                pg.draw.rect(self.screen, self.CARD_COLOR, card, border_radius=8)
                # Checker pattern background for transparency
                self._draw_checker(face_rect)
                self.screen.blit(face_surf, face_rect)

        # Buttons
        self._draw_button("Retake", self.screen_w//2 - 150, 500, 140, 50,
                          self.RED, self.DARK_RED)
        self._draw_button("Manual Select", self.screen_w//2, 500, 150, 50,
                          self.BLUE, self.LIGHT_BLUE)
        self._draw_button("Confirm", self.screen_w//2 + 150, 500, 140, 50,
                          self.GREEN, self.DARK_GREEN)

    def _enter_manual_select(self):
        """Prepare and enter manual face selection mode."""
        if self.face_image is None:
            return
        self.state = 'manual_select'
        self._select_start = None
        self._select_end = None
        self._selecting = False
        # Prepare scaled photo surface
        max_w = self.screen_w - 40
        max_h = self.screen_h - 120
        h, w = self.face_image.shape[:2]
        self._manual_scale = min(max_w / w, max_h / h)
        new_w = int(w * self._manual_scale)
        new_h = int(h * self._manual_scale)
        resized = cv2.resize(self.face_image, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._manual_photo_surface = pg.image.frombuffer(rgb.tobytes(), (new_w, new_h), 'RGB')
        self._manual_photo_rect = self._manual_photo_surface.get_rect(
            center=(self.screen_w // 2, (self.screen_h - 60) // 2 + 20))

    def _draw_manual_select(self):
        """Draw the manual face selection interface."""
        # Title
        title = self.label_font.render("拖拽选取人脸区域", True, self.LIGHT_BLUE)
        title_rect = title.get_rect(center=(self.screen_w // 2, 15))
        self.screen.blit(title, title_rect)

        # Draw photo
        if self._manual_photo_surface and self._manual_photo_rect:
            self.screen.blit(self._manual_photo_surface, self._manual_photo_rect)

        # Draw selection rectangle
        if self._select_start and self._select_end:
            sx, sy = self._select_start
            ex, ey = self._select_end
            rect = pg.Rect(min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy))
            if self._selecting:
                # Dashed rectangle while dragging
                self._draw_dashed_rect(self.screen, self.YELLOW, rect)
            else:
                # Solid rectangle when confirmed
                pg.draw.rect(self.screen, self.GREEN, rect, 2)

        # Buttons at bottom
        btn_y = self.screen_h - 30
        self._draw_button("Confirm", self.screen_w//2 + 80, btn_y, 140, 45,
                          self.GREEN, self.DARK_GREEN)
        self._draw_button("Back", self.screen_w//2 - 80, btn_y, 140, 45,
                          self.DARK_GRAY, self.GRAY)

    def _handle_manual_select_events(self, event):
        """Handle mouse events for manual face selection."""
        x, y = event.pos
        # Check button clicks
        btn_y = self.screen_h - 30
        if self._is_in_button(x, y, self.screen_w//2 + 80, btn_y, 140, 45):
            # Confirm: extract face with manual bbox
            self._confirm_manual_select()
            return
        if self._is_in_button(x, y, self.screen_w//2 - 80, btn_y, 140, 45):
            # Back
            self.state = 'preview' if self.face_rgba else 'main_menu'
            return
        # Start selection drag
        if self._manual_photo_rect and self._manual_photo_rect.collidepoint(x, y):
            self._select_start = (x, y)
            self._select_end = (x, y)
            self._selecting = True

    def _confirm_manual_select(self):
        """Use manual selection bbox to extract face."""
        if self._select_start is None or self._select_end is None:
            return
        if self.face_image is None or self._manual_photo_rect is None:
            return

        sx, sy = self._select_start
        ex, ey = self._select_end
        # Convert screen coords to photo coords
        px = self._manual_photo_rect.x
        py = self._manual_photo_rect.y
        inv = 1.0 / self._manual_scale
        x1 = int((min(sx, ex) - px) * inv)
        y1 = int((min(sy, ey) - py) * inv)
        x2 = int((max(sx, ex) - px) * inv)
        y2 = int((max(sy, ey) - py) * inv)
        # Clamp to image bounds
        h, w = self.face_image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        bbox = (x1, y1, x2 - x1, y2 - y1)
        face = self.detector.extract_face(self.face_image, bbox=bbox)
        if face is not None:
            self.face_rgba = face
            self._style_cache.clear()
            self.state = 'preview'
        else:
            # Fallback: manual crop with ellipse mask
            region = self.face_image[y1:y2, x1:x2].copy()
            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            center = (region.shape[1] // 2, region.shape[0] // 2)
            axes = (region.shape[1] // 2 - 2, region.shape[0] // 2 - 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 3)
            rgba = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            self.face_rgba = rgba
            self._style_cache.clear()
            self.state = 'preview'

    def _draw_dashed_rect(self, surface, color, rect):
        """Draw a dashed rectangle."""
        dash_len = 8
        gap_len = 5
        lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),
            ((rect.right, rect.top), (rect.right, rect.bottom)),
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),
            ((rect.left, rect.bottom), (rect.left, rect.top)),
        ]
        for (x1, y1), (x2, y2) in lines:
            length = math.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            dx, dy = (x2 - x1) / length, (y2 - y1) / length
            pos = 0.0
            drawing = True
            while pos < length:
                seg_len = dash_len if drawing else gap_len
                end_pos = min(pos + seg_len, length)
                if drawing:
                    p1 = (int(x1 + dx * pos), int(y1 + dy * pos))
                    p2 = (int(x1 + dx * end_pos), int(y1 + dy * end_pos))
                    pg.draw.line(surface, color, p1, p2, 2)
                pos = end_pos
                drawing = not drawing

    def _draw_style_select(self):
        """Draw the style selection screen with 2 style cards."""
        title = self.title_font.render("Choose Style", True, self.GOLD)
        title_rect = title.get_rect(center=(self.screen_w // 2, 50))
        self.screen.blit(title, title_rect)

        subtitle = self.label_font.render("Click a style to preview, then Start Game!", True, self.GRAY)
        sub_rect = subtitle.get_rect(center=(self.screen_w // 2, 110))
        self.screen.blit(subtitle, sub_rect)

        # Style cards - 2 styles centered
        styles = [
            ('sprite', 'Sprite Art'),
            ('original', 'Original'),
        ]

        start_x = self.screen_w // 2 - 185  # Center 2 cards with 220px spacing
        for i, (style_key, style_label) in enumerate(styles):
            sx = start_x + i * 220
            sy = 160

            # Card background
            is_selected = (self.selected_style == style_key)
            card_color = self.BLUE if is_selected else self.CARD_COLOR
            border_color = self.GOLD if is_selected else self.DARK_GRAY
            card_rect = pg.Rect(sx, sy, 150, 230)
            pg.draw.rect(self.screen, card_color, card_rect, border_radius=10)
            pg.draw.rect(self.screen, border_color, card_rect, 3, border_radius=10)

            # Style preview image (from cache or generate)
            is_loading_this = (self._loading and self._loading_style == style_key)
            if is_loading_this:
                # Show loading animation
                now = pg.time.get_ticks()
                if now - self._loading_timer > 500:
                    self._loading_dots = (self._loading_dots + 1) % 4
                    self._loading_timer = now
                dots = "." * self._loading_dots
                loading_text = self.label_font.render(f"Generating{dots}", True, self.GOLD)
                loading_rect = loading_text.get_rect(center=(sx + 75, sy + 100))
                self.screen.blit(loading_text, loading_rect)
            elif style_key in self._style_cache:
                preview = self._cv2_to_pg_surface(self._style_cache[style_key], 120)
                if preview:
                    preview_rect = preview.get_rect(center=(sx + 75, sy + 100))
                    self._draw_checker(preview_rect)
                    self.screen.blit(preview, preview_rect)
            elif self.face_rgba is not None:
                # For original style, generate on the fly (fast)
                if style_key == 'original':
                    styled = apply_style(self.face_rgba, style_key)
                    self._style_cache[style_key] = styled
                    preview = self._cv2_to_pg_surface(styled, 120)
                    if preview:
                        preview_rect = preview.get_rect(center=(sx + 75, sy + 100))
                        self._draw_checker(preview_rect)
                        self.screen.blit(preview, preview_rect)

            # Label
            label = self.button_font.render(style_label, True, self.WHITE)
            label_rect = label.get_rect(center=(sx + 75, sy + 205))
            self.screen.blit(label, label_rect)

            # Selected indicator
            if is_selected:
                check = self.button_font.render("*", True, self.GOLD)
                self.screen.blit(check, (sx + 5, sy + 5))

        # Large preview of selected style
        if self.styled_face is not None and not self._loading:
            label = self.label_font.render(f"Preview: {self.selected_style}", True, self.LIGHT_BLUE)
            label_rect = label.get_rect(center=(self.screen_w // 2, 420))
            self.screen.blit(label, label_rect)

        # Loading indicator below cards
        if self._loading:
            now = pg.time.get_ticks()
            if now - self._loading_timer > 500:
                self._loading_dots = (self._loading_dots + 1) % 4
                self._loading_timer = now
            dots = "." * self._loading_dots
            loading = self.label_font.render(f"Generating sprite{dots}", True, self.GOLD)
            loading_rect = loading.get_rect(center=(self.screen_w // 2, 430))
            self.screen.blit(loading, loading_rect)

        # Buttons
        self._draw_button("Back", self.screen_w//2 - 200, 500, 120, 50,
                          self.DARK_GRAY, self.GRAY)

        if self._loading:
            # Grayed out button while loading
            self._draw_button("Generating...", self.screen_w//2, 500, 280, 60,
                              self.DARK_GRAY)
        else:
            self._draw_button("Start Game!", self.screen_w//2, 500, 280, 60,
                              self.GREEN, self.DARK_GREEN)

    def _draw_checker(self, rect):
        """Draw a checker pattern to show transparency."""
        checker_size = 8
        colors = [(200, 200, 200), (240, 240, 240)]
        for y in range(rect.top, rect.bottom, checker_size):
            for x in range(rect.left, rect.right, checker_size):
                ci = ((x - rect.left) // checker_size + (y - rect.top) // checker_size) % 2
                w = min(checker_size, rect.right - x)
                h = min(checker_size, rect.bottom - y)
                pg.draw.rect(self.screen, colors[ci], (x, y, w, h))
