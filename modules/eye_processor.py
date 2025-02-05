import cv2
import numpy as np
from tensorflow import keras
from typing import Tuple

class EyeProcessor:
    def __init__(self, model_path: str, left_cascade_path: str, right_cascade_path: str):
        self.model = keras.models.load_model(model_path)
        self.left_eye_detector = cv2.CascadeClassifier(left_cascade_path)
        self.right_eye_detector = cv2.CascadeClassifier(right_cascade_path)

    def process_eye(self, eye_frame: np.ndarray) -> str:
        """
        Procesa un frame de ojo y determina si está abierto o cerrado.
        """
        eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (24, 24))
        eye = eye / 255.0
        eye = eye.reshape(24, 24, -1)
        eye = np.expand_dims(eye, axis=0)
        prediction = self.model.predict(eye)
        return "Open" if np.argmax(prediction, axis=1)[0] == 1 else "Closed"

    def process_left_eye(self, face_frame: np.ndarray, face_gray: np.ndarray) -> str:
        """
        Detecta y procesa el ojo izquierdo.
        """
        return self._process_eye_region(face_frame, face_gray, self.left_eye_detector)

    def process_right_eye(self, face_frame: np.ndarray, face_gray: np.ndarray) -> str:
        """
        Detecta y procesa el ojo derecho.
        """
        return self._process_eye_region(face_frame, face_gray, self.right_eye_detector)

    def _process_eye_region(self, face_frame: np.ndarray, face_gray: np.ndarray, 
                          detector: cv2.CascadeClassifier) -> str:
        """
        Procesa una región de ojo usando el detector especificado.
        """
        eyes = detector.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_frame = face_frame[ey:ey+eh, ex:ex+ew]
            return self.process_eye(eye_frame)
        return "Unknown"
