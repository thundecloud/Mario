"""
Face detection and extraction module using MediaPipe Tasks API.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

# Path to model files
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_FACE_DETECTOR_MODEL = os.path.join(_MODELS_DIR, 'blaze_face_short_range.tflite')
_FACE_LANDMARKER_MODEL = os.path.join(_MODELS_DIR, 'face_landmarker.task')


def _download_models():
    """Download model files if they don't exist."""
    import urllib.request
    os.makedirs(_MODELS_DIR, exist_ok=True)

    if not os.path.exists(_FACE_DETECTOR_MODEL):
        url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
        urllib.request.urlretrieve(url, _FACE_DETECTOR_MODEL)

    if not os.path.exists(_FACE_LANDMARKER_MODEL):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        urllib.request.urlretrieve(url, _FACE_LANDMARKER_MODEL)


class FaceDetector:
    """Detects faces and extracts face regions using MediaPipe Tasks API."""

    def __init__(self):
        _download_models()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Face detector
        det_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_FACE_DETECTOR_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(det_options)

        # Face landmarker
        lm_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_LANDMARKER_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(lm_options)

    def detect_face_bbox(self, image):
        """Detect face bounding box in BGR image.
        Returns (x, y, w, h) or None if no face found."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_detector.detect(mp_image)

        if not result.detections:
            return None

        # Take the detection with highest confidence
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box
        h, w = image.shape[:2]

        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        bw = min(bbox.width, w - x)
        bh = min(bbox.height, h - y)

        return (x, y, bw, bh)

    def get_face_landmarks(self, image):
        """Get face mesh landmarks.
        Returns list of (x, y) pixel coordinates or None."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = []
        for lm in result.face_landmarks[0]:
            landmarks.append((int(lm.x * w), int(lm.y * h)))

        return landmarks

    def extract_face(self, image, padding=0.3):
        """Extract face region with alpha mask from image.
        Returns RGBA numpy array or None if no face detected.
        padding: extra space around face as fraction of face size."""
        bbox = self.detect_face_bbox(image)
        if bbox is None:
            return None

        x, y, bw, bh = bbox
        h, w = image.shape[:2]

        # Add padding
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        face_region = image[y1:y2, x1:x2].copy()

        # Get landmarks for precise mask
        landmarks = self.get_face_landmarks(image)

        if landmarks is not None:
            # Create convex hull mask from face oval landmarks
            face_oval_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]

            oval_points = []
            for idx in face_oval_indices:
                if idx < len(landmarks):
                    px, py = landmarks[idx]
                    oval_points.append((px - x1, py - y1))

            if oval_points:
                mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
                hull = cv2.convexHull(np.array(oval_points, dtype=np.int32))
                cv2.fillConvexPoly(mask, hull, 255)

                # Smooth mask edges
                mask = cv2.GaussianBlur(mask, (7, 7), 3)

                # Convert to RGBA
                rgba = cv2.cvtColor(face_region, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask
                return rgba

        # Fallback: elliptical mask
        mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
        center = (face_region.shape[1] // 2, face_region.shape[0] // 2)
        axes = (face_region.shape[1] // 2 - 2, face_region.shape[0] // 2 - 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 3)

        rgba = cv2.cvtColor(face_region, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        return rgba

    def close(self):
        """Release resources."""
        self.face_detector.close()
        self.face_landmarker.close()
