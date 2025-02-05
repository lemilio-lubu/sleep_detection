import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from .eye_processor import EyeProcessor

class FaceDetector:
    def __init__(self, cascade_path: str, eye_processor: EyeProcessor):
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.eye_processor = eye_processor

    def detect_faces(self, frame: np.ndarray, min_neighbors: int = 5,
                    scale_factor: float = 1.1, min_size: Tuple[int, int] = (25, 25)) -> List[Dict[str, Any]]:
        """
        Detecta rostros en un frame y analiza el estado de los ojos.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            minNeighbors=min_neighbors,
            scaleFactor=scale_factor,
            minSize=min_size
        )

        return [self._analyze_face(frame, gray, face) for face in faces]

    def _analyze_face(self, frame: np.ndarray, gray: np.ndarray, 
                     face_coords: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analiza un rostro detectado para determinar el estado de los ojos.
        """
        x, y, w, h = face_coords
        face_frame = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        left_status = self.eye_processor.process_left_eye(face_frame, face_gray)
        right_status = self.eye_processor.process_right_eye(face_frame, face_gray)

        return {
            'face_location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'left_eye_status': left_status,
            'right_eye_status': right_status,
            'is_active': not (left_status == "Closed" and right_status == "Closed")
        }
