import cv2
import numpy as np
import logging
import traceback
from typing import Optional, Tuple, Any
from .alert_manager import AlertManager
from .ear_calculator import calculate_ear

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, face_mesh, left_eye_indices, right_eye_indices):
        """
        Inicializa el procesador de frames.
        
        Args:
            face_mesh: Detector de landmarks faciales de MediaPipe
            left_eye_indices: Índices de landmarks del ojo izquierdo
            right_eye_indices: Índices de landmarks del ojo derecho
        """
        self.alert_manager = AlertManager()
        self.face_mesh = face_mesh
        self.LEFT_EYE = left_eye_indices
        self.RIGHT_EYE = right_eye_indices
        self.JPEG_QUALITY = 80

    async def process_frame(self, frame_data: bytes, connection_data: dict) -> Tuple[bytes, Optional[dict]]:
        """
        Procesa un frame y retorna el resultado y posibles alertas.
        
        Args:
            frame_data: Frame en formato bytes
            connection_data: Datos de la conexión actual
            
        Returns:
            Tuple[bytes, Optional[dict]]: Frame procesado y alerta si existe
        """
        try:
            frame = self._decode_frame(frame_data)
            if frame is None:
                return frame_data, None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb_frame)

            if not face_results.multi_face_landmarks:
                return self._encode_frame(frame), None

            return self._process_face_landmarks(frame, face_results.multi_face_landmarks[0], connection_data)

        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            logger.error(traceback.format_exc())
            return frame_data, None

    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Decodifica los datos del frame a una imagen."""
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("No se pudo decodificar el frame")
            return frame
        except Exception as e:
            logger.error(f"Error decodificando frame: {e}")
            return None

    def _process_face_landmarks(self, frame: np.ndarray, face_landmarks: Any, 
                              connection_data: dict) -> Tuple[bytes, Optional[dict]]:
        """
        Procesa los landmarks faciales y genera alertas si es necesario.
        
        Args:
            frame: Frame de video
            face_landmarks: Landmarks faciales detectados
            connection_data: Datos de la conexión
            
        Returns:
            Tuple[bytes, Optional[dict]]: Frame procesado y alerta si existe
        """
        left_eye_points = [face_landmarks.landmark[i] for i in self.LEFT_EYE]
        right_eye_points = [face_landmarks.landmark[i] for i in self.RIGHT_EYE]
        
        avg_ear = self._calculate_average_ear(left_eye_points, right_eye_points)
        self._draw_landmarks(frame, face_landmarks)
        self._draw_ear_info(frame, avg_ear)
        
        alert_info, is_critical = self.alert_manager.check_alert_status(avg_ear, connection_data)
        
        if is_critical or alert_info:
            self._draw_alert_box(frame, is_critical)
            self._draw_alert_text(frame, alert_info)
        
        return self._encode_frame(frame), alert_info

    def _calculate_average_ear(self, left_eye_points: list, right_eye_points: list) -> float:
        """Calcula el EAR promedio de ambos ojos."""
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        return (left_ear + right_ear) / 2

    def _draw_landmarks(self, frame: np.ndarray, face_landmarks: Any) -> None:
        """Dibuja los landmarks faciales en el frame."""
        height, width = frame.shape[:2]
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def _draw_ear_info(self, frame: np.ndarray, ear: float) -> None:
        """Dibuja la información del EAR en el frame."""
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_alert_box(self, frame: np.ndarray, is_critical: bool) -> None:
        """Dibuja el rectángulo de alerta en el frame."""
        height, width = frame.shape[:2]
        color = (0, 0, 255) if is_critical else (0, 255, 255)
        thickness = 10 if is_critical else 5
        cv2.rectangle(frame, (0, 0), (width, height), color, thickness)

    def _draw_alert_text(self, frame: np.ndarray, alert_info: dict) -> None:
        """Dibuja el texto de alerta en el frame."""
        if alert_info:
            cv2.putText(frame, alert_info["message"], (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Tiempo: {alert_info['elapsed_time']:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Codifica el frame procesado a bytes."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        return buffer.tobytes()

    def set_jpeg_quality(self, quality: int) -> None:
        """Establece la calidad de compresión JPEG."""
        self.JPEG_QUALITY = max(1, min(100, quality))
