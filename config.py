from typing import List, Tuple

class FaceMeshConfig:
    MAX_FACES: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5

class EyeIndices:
    LEFT_EYE: List[int] = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE: List[int] = [33, 160, 158, 133, 153, 144]

class AlertConfig:
    INITIAL_THRESHOLD: float = 2.0
    DANGER_THRESHOLD: float = 3.0
    EYE_AR_THRESHOLD: float = 0.2
    MAX_FRAME_SIZE: int = 1024 * 1024

class ModelPaths:
    FACE_CASCADE: str = 'haar/haarcascade_frontalface_alt.xml'
    LEFT_EYE_CASCADE: str = 'haar/haarcascade_lefteye_2splits.xml'
    RIGHT_EYE_CASCADE: str = 'haar/haarcascade_righteye_2splits.xml'
    SLEEP_MODEL: str = 'modelos/sleep_model.h5'
