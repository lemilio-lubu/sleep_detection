import logging
import traceback  # Añadir esta importación
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
from typing import Dict, Optional, Tuple, Any
import time
import json

# Inicializar la aplicación FastAPI
app = FastAPI()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices de los landmarks para los ojos
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Configuración
ALERT_THRESHOLD = 2.0  # Segundos para activar la alerta inicial
DANGER_THRESHOLD = 3.0  # Segundos para alerta crítica
EYE_AR_THRESH = 0.2  # Umbral para determinar si el ojo está cerrado
MAX_FRAME_SIZE = 1024 * 1024  # 1MB máximo por frame

def calculate_ear(eye_points):
    """Calcula el Eye Aspect Ratio (EAR)"""
    try:
        # Convertir landmarks a coordenadas numéricas
        points = [(point.x, point.y) for point in eye_points]
        
        # Calcular distancias verticales
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        
        # Calcular distancia horizontal
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        # Calcular EAR
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except Exception as e:
        logger.error(f"Error calculando EAR: {e}")
        return 0.3  # Valor por defecto que indica ojos abiertos

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = {
            'closed_eyes_start_time': None,
            'last_frame_time': time.time(),
            'alert_level': 0,
            'critical_alert_active': False  # Solo para alertas críticas
        }
        logger.info("Nueva conexión establecida")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            logger.info("Conexión cerrada")

manager = ConnectionManager()

class AlertManager:
    def __init__(self):
        self.ALERT_THRESHOLD = 2.0
        self.DANGER_THRESHOLD = 3.0
        self.EYE_AR_THRESH = 0.2

    def check_alert_status(self, avg_ear: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """Determina el estado de alerta basado en EAR."""
        if avg_ear >= self.EYE_AR_THRESH:
            self._reset_alert_status(connection_data)
            return None, False

        if connection_data['closed_eyes_start_time'] is None:
            connection_data['closed_eyes_start_time'] = time.time()

        elapsed_time = time.time() - connection_data['closed_eyes_start_time']
        
        if elapsed_time > self.DANGER_THRESHOLD:
            return self._handle_danger_alert(elapsed_time, connection_data)
        
        if elapsed_time > self.ALERT_THRESHOLD:
            return self._handle_warning_alert(elapsed_time, connection_data)
        
        return None, False

    def _reset_alert_status(self, connection_data: dict) -> None:
        """Reinicia el estado de alerta."""
        connection_data['closed_eyes_start_time'] = None
        if connection_data.get('alert_level', 0) == 1:
            connection_data['alert_level'] = 0

    def _handle_danger_alert(self, elapsed_time: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """Maneja alertas de nivel crítico."""
        if connection_data['critical_alert_active']:
            return None, True

        alert_info = {
            "level": 2,
            "message": "¡PELIGRO! Somnolencia crítica detectada",
            "elapsed_time": elapsed_time
        }
        connection_data['critical_alert_active'] = True
        connection_data['alert_level'] = 2
        return alert_info, True

    def _handle_warning_alert(self, elapsed_time: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """Maneja alertas de advertencia."""
        if connection_data['critical_alert_active']:
            return None, False

        return {
            "level": 1,
            "message": "Advertencia: Signos de somnolencia",
            "elapsed_time": elapsed_time
        }, True

class FrameProcessor:
    def __init__(self):
        self.alert_manager = AlertManager()

    async def process_frame(self, frame_data: bytes, connection_data: dict) -> Tuple[bytes, Optional[dict]]:
        """Procesa un frame y retorna el resultado y posibles alertas."""
        try:
            frame = self._decode_frame(frame_data)
            if frame is None:
                return frame_data, None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)

            if not face_results.multi_face_landmarks:
                return self._encode_frame(frame), None

            return self._process_face_landmarks(frame, face_results.multi_face_landmarks[0], connection_data)

        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            logger.error(traceback.format_exc())  # Ahora funcionará correctamente
            return frame_data, None

    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Decodifica los datos del frame a una imagen."""
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decodificando frame: {e}")
            return None

    def _process_face_landmarks(self, frame: np.ndarray, face_landmarks: Any, 
                              connection_data: dict) -> Tuple[bytes, Optional[dict]]:
        """Procesa los landmarks faciales y genera alertas si es necesario."""
        left_eye_points = [face_landmarks.landmark[i] for i in LEFT_EYE]
        right_eye_points = [face_landmarks.landmark[i] for i in RIGHT_EYE]
        
        avg_ear = self._calculate_average_ear(left_eye_points, right_eye_points)
        self._draw_ear_info(frame, avg_ear)
        
        alert_info, is_critical = self.alert_manager.check_alert_status(avg_ear, connection_data)
        
        if is_critical or alert_info:
            self._draw_alert_box(frame, is_critical)
        
        return self._encode_frame(frame), alert_info

    def _calculate_average_ear(self, left_eye_points: list, right_eye_points: list) -> float:
        """Calcula el EAR promedio de ambos ojos."""
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        return (left_ear + right_ear) / 2

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

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Codifica el frame procesado a bytes."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

class WebSocketHandler:
    def __init__(self):
        self.frame_processor = FrameProcessor()
        self.MAX_FRAME_SIZE = 1024 * 1024

    async def handle_message(self, message: dict, websocket: WebSocket, 
                           connection_data: dict) -> None:
        """Maneja los mensajes recibidos por el WebSocket."""
        if self._is_reset_command(message):
            await self._handle_reset(websocket, connection_data)
            return

        if not self._is_valid_frame(message):
            return

        await self._process_and_send_frame(message["bytes"], websocket, connection_data)

    def _is_reset_command(self, message: dict) -> bool:
        """Verifica si el mensaje es un comando de reset."""
        return message.get("bytes") and len(message["bytes"]) == 1 and message["bytes"][0] == 0

    def _is_valid_frame(self, message: dict) -> bool:
        """Verifica si el mensaje contiene un frame válido."""
        return "bytes" in message and len(message["bytes"]) <= self.MAX_FRAME_SIZE

    async def _handle_reset(self, websocket: WebSocket, connection_data: dict) -> None:
        """Maneja el comando de reset."""
        self._reset_connection_state(connection_data)
        await self._send_reset_confirmation(websocket)

    def _reset_connection_state(self, connection_data: dict) -> None:
        """Reinicia el estado de la conexión."""
        connection_data.update({
            'closed_eyes_start_time': None,
            'alert_level': 0,
            'critical_alert_active': False
        })

    async def _send_reset_confirmation(self, websocket: WebSocket) -> None:
        """Envía confirmación de reset al cliente."""
        reset_confirm = {
            "type": "reset_confirm",
            "message": "Estado reiniciado"
        }
        await websocket.send_text(json.dumps(reset_confirm))

    async def _process_and_send_frame(self, frame_data: bytes, websocket: WebSocket, 
                                    connection_data: dict) -> None:
        """Procesa y envía el frame al cliente."""
        processed_frame, alert = await self.frame_processor.process_frame(frame_data, connection_data)
        await websocket.send_bytes(processed_frame)
        
        if alert:
            await websocket.send_text(json.dumps(alert))
        
        connection_data['last_frame_time'] = time.time()

# Inicializar componentes
websocket_handler = WebSocketHandler()

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive()
            await websocket_handler.handle_message(
                message, 
                websocket, 
                manager.active_connections[websocket]
            )
    except WebSocketDisconnect:
        logger.info("Cliente desconectado normalmente")
    except Exception as e:
        logger.error(f"Error en la conexión: {e}")
        logger.error(traceback.format_exc())  # Ahora funcionará correctamente
    finally:
        manager.disconnect(websocket)