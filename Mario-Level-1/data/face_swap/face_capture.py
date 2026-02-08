"""
Face capture module - handles camera capture and photo upload.
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image


def capture_from_camera():
    """Opens camera, shows preview, captures photo on spacebar press.
    Returns BGR numpy array or None if cancelled."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None

    captured = None
    cv2.namedWindow("Camera - Press SPACE to capture, ESC to cancel", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for natural selfie feel
        display = cv2.flip(frame, 1)
        cv2.imshow("Camera - Press SPACE to capture, ESC to cancel", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            captured = cv2.flip(frame, 1)  # Save mirrored version
            break
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


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


def get_face_image(method='camera'):
    """Main entry point for face capture.
    method: 'camera' or 'upload'
    Returns BGR numpy array or None."""
    if method == 'camera':
        return capture_from_camera()
    elif method == 'upload':
        return upload_photo()
    else:
        raise ValueError(f"Unknown method: {method}")
