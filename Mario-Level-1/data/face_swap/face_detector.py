"""
Face detection and extraction module using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """Detects faces and extracts face regions using MediaPipe."""

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect_face_bbox(self, image):
        """Detect face bounding box in BGR image.
        Returns (x, y, w, h) or None if no face found."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if not results.detections:
            return None

        # Take the detection with highest confidence
        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        h, w = image.shape[:2]

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        return (x, y, bw, bh)

    def get_face_landmarks(self, image):
        """Get 478 face mesh landmarks.
        Returns list of (x, y) pixel coordinates or None."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
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
            # Face oval indices in MediaPipe
            face_oval_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]

            oval_points = []
            for idx in face_oval_indices:
                if idx < len(landmarks):
                    px, py = landmarks[idx]
                    # Adjust coordinates to face_region
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
        self.face_detection.close()
        self.face_mesh.close()
