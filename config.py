from pathlib import Path

class ModelPaths:
    BASE_DIR = Path(__file__).parent
    FACE_CASCADE = str(BASE_DIR / 'haar/haarcascade_frontalface_alt.xml')
    LEFT_EYE_CASCADE = str(BASE_DIR / 'haar/haarcascade_lefteye_2splits.xml')
    RIGHT_EYE_CASCADE = str(BASE_DIR / 'haar/haarcascade_righteye_2splits.xml')
    SLEEP_MODEL = str(BASE_DIR / 'modelos/sleep_model.h5')

class DetectionConfig:
    MIN_FACE_SIZE = (25, 25)
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5
    EYE_SIZE = (24, 24)
