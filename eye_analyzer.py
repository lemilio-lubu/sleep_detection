import cv2
import numpy as np
from typing import Tuple, Dict, Any
from tensorflow import keras
from config import AlertConfig, ModelPaths

class EyeAnalyzer:
    def __init__(self):
        self.model = keras.models.load_model(ModelPaths.SLEEP_MODEL)
        self.face_detector = cv2.CascadeClassifier(ModelPaths.FACE_CASCADE)
        self.left_eye_detector = cv2.CascadeClassifier(ModelPaths.LEFT_EYE_CASCADE)
        self.right_eye_detector = cv2.CascadeClassifier(ModelPaths.RIGHT_EYE_CASCADE)

    def process_eye(self, eye_frame: np.ndarray) -> int:
        """Process eye frame and predict if it's open or closed."""
        eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (24, 24))
        eye = eye / 255.0
        eye = eye.reshape(24, 24, -1)
        eye = np.expand_dims(eye, axis=0)
        prediction = self.model.predict(eye)
        return int(np.argmax(prediction, axis=1)[0])

    def analyze_face(self, frame: np.ndarray, gray: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analyze a face region and detect eye states."""
        x, y, w, h = face_coords
        face_frame = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        left_eyes = self.left_eye_detector.detectMultiScale(face_gray)
        right_eyes = self.right_eye_detector.detectMultiScale(face_gray)

        left_eye_status = self._process_eye_region(face_frame, left_eyes)
        right_eye_status = self._process_eye_region(face_frame, right_eyes)
        
        return {
            'face_location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'left_eye_status': left_eye_status,
            'right_eye_status': right_eye_status,
            'is_active': not (left_eye_status == "Closed" and right_eye_status == "Closed")
        }

    def _process_eye_region(self, face_frame: np.ndarray, eye_regions: np.ndarray) -> str:
        """Process detected eye regions and determine if eye is open or closed."""
        for (ex, ey, ew, eh) in eye_regions:
            eye_frame = face_frame[ey:ey+eh, ex:ex+ew]
            prediction = self.process_eye(eye_frame)
            return "Open" if prediction == 1 else "Closed"
        return "Unknown"
