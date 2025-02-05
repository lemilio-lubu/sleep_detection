import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ear(eye_points):
    """Calcula el Eye Aspect Ratio (EAR)"""
    try:
        points = [(point.x, point.y) for point in eye_points]
        
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except Exception as e:
        logger.error(f"Error calculando EAR: {e}")
        return 0.3
