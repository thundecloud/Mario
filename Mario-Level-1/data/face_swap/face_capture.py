"""
Face capture module - handles camera capture (Pygame-embedded) and photo upload.
"""

import cv2
import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog

import pygame as pg


class CameraCapture:
    """Pygame-based camera capture with face guide overlay and real-time detection."""

    # Colors
    WHITE = (255, 255, 255)
    GREEN = (70, 200, 100)
    RED = (200, 70, 70)
    YELLOW = (255, 215, 0)
    DARK_BG = (40, 44, 52)
    GUIDE_COLOR = (200, 200, 200)

    def __init__(self, screen, detector):
        self.screen = screen
        self.detector = detector
        self.screen_w, self.screen_h = screen.get_size()

        # Font setup
        self.font = pg.font.SysFont('Microsoft YaHei', 24)
        self.small_font = pg.font.SysFont('Microsoft YaHei', 18)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        """Main loop. Returns (image_bgr, face_bbox) or (None, None) if cancelled."""
        clock = pg.time.Clock()
        timestamp = 0
        face_bbox = None
        stable_frames = 0
        last_frame = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # Mirror
            last_frame = frame

            # Real-time fast detection
            timestamp += 33  # ~30fps
            bbox = self.detector.detect_face_fast(frame, timestamp)

            if bbox:
                face_bbox = bbox
                stable_frames += 1
            else:
                face_bbox = None
                stable_frames = 0

            # Render to Pygame
            self._draw_frame(frame, bbox, stable_frames)

            # Event handling
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.cap.release()
                    return (None, None)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE and face_bbox is not None:
                        self.cap.release()
                        return (last_frame, face_bbox)
                    elif event.key == pg.K_ESCAPE:
                        self.cap.release()
                        return (None, None)

            clock.tick(30)

    def _draw_frame(self, frame, bbox, stable_frames):
        """Draw camera frame + guide overlay + detection feedback."""
        self.screen.fill(self.DARK_BG)

        # Convert OpenCV BGR frame to Pygame surface
        h, w = frame.shape[:2]
        # Scale frame to fit screen while maintaining aspect ratio
        scale = min(self.screen_w / w, (self.screen_h - 80) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        surface = pg.image.frombuffer(rgb.tobytes(), (new_w, new_h), 'RGB')

        # Center the camera view
        offset_x = (self.screen_w - new_w) // 2
        offset_y = 10
        self.screen.blit(surface, (offset_x, offset_y))

        # Draw dashed ellipse guide (centered on camera view)
        guide_cx = offset_x + new_w // 2
        guide_cy = offset_y + new_h // 2
        guide_rx = int(new_w * 0.2)
        guide_ry = int(new_h * 0.35)
        guide_rect = pg.Rect(guide_cx - guide_rx, guide_cy - guide_ry,
                             guide_rx * 2, guide_ry * 2)
        self._draw_dashed_ellipse(self.screen, self.GUIDE_COLOR, guide_rect)

        # Draw detection bbox if found
        if bbox:
            bx, by, bw, bh = bbox
            # Scale bbox to display coordinates
            dx = int(bx * scale) + offset_x
            dy = int(by * scale) + offset_y
            dw = int(bw * scale)
            dh = int(bh * scale)
            pg.draw.rect(self.screen, self.GREEN, (dx, dy, dw, dh), 2)

        # Status text
        status_y = offset_y + new_h + 10
        if bbox is None:
            status_text = self.font.render("未检测到人脸，请调整位置", True, self.RED)
        elif stable_frames < 5:
            status_text = self.font.render("已检测到人脸，请保持不动...", True, self.YELLOW)
        else:
            status_text = self.font.render("已检测到人脸！按空格拍照", True, self.GREEN)
        status_rect = status_text.get_rect(center=(self.screen_w // 2, status_y))
        self.screen.blit(status_text, status_rect)

        # Bottom hint
        hint = self.small_font.render("空格 = 拍照  |  ESC = 取消", True, self.GUIDE_COLOR)
        hint_rect = hint.get_rect(center=(self.screen_w // 2, self.screen_h - 20))
        self.screen.blit(hint, hint_rect)

        pg.display.flip()

    def _draw_dashed_ellipse(self, surface, color, rect):
        """Draw a dashed ellipse outline."""
        cx, cy = rect.center
        rx, ry = rect.width // 2, rect.height // 2
        num_segments = 60
        for i in range(0, num_segments, 2):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi
            p1 = (cx + int(rx * math.cos(angle1)), cy + int(ry * math.sin(angle1)))
            p2 = (cx + int(rx * math.cos(angle2)), cy + int(ry * math.sin(angle2)))
            pg.draw.line(surface, color, p1, p2, 2)


def upload_photo():
    """Opens file dialog to select a photo. Returns BGR numpy array or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select a photo",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()

    if not file_path or not os.path.exists(file_path):
        return None

    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Cannot read image: {file_path}")
        return None

    return image
