import logging
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
from typing import Dict, Optional
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

async def process_frame(frame_data: bytes, connection_data: dict) -> tuple[bytes, Optional[dict]]:
    try:
        # Decodificar frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar el frame")

        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        alert_info = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener puntos de los ojos
                left_eye_points = [face_landmarks.landmark[i] for i in LEFT_EYE]
                right_eye_points = [face_landmarks.landmark[i] for i in RIGHT_EYE]
                
                # Calcular EAR para ambos ojos
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2
                
                # Dibujar información en el frame
                height, width = frame.shape[:2]
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Lógica de alertas mejorada
                if avg_ear < EYE_AR_THRESH:
                    if connection_data['closed_eyes_start_time'] is None:
                        connection_data['closed_eyes_start_time'] = time.time()
                    
                    elapsed_time = time.time() - connection_data['closed_eyes_start_time']
                    
                    # Siempre mostrar el tiempo transcurrido
                    cv2.putText(frame, f"Tiempo: {elapsed_time:.1f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 0, 255) if elapsed_time > ALERT_THRESHOLD else (0, 255, 0), 2)
                    
                    if elapsed_time > DANGER_THRESHOLD:
                        # Alerta crítica - requiere confirmación del usuario
                        if not connection_data['critical_alert_active']:
                            alert_info = {
                                "level": 2,
                                "message": "¡PELIGRO! Somnolencia crítica detectada",
                                "elapsed_time": elapsed_time
                            }
                            connection_data['critical_alert_active'] = True
                            connection_data['alert_level'] = 2
                        # Mantener el marco rojo mientras la alerta crítica esté activa
                        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
                    
                    elif elapsed_time > ALERT_THRESHOLD:
                        # Alerta visual leve - no requiere confirmación
                        if not connection_data['critical_alert_active']:  # No mostrar si hay alerta crítica activa
                            alert_info = {
                                "level": 1,
                                "message": "Advertencia: Signos de somnolencia",
                                "elapsed_time": elapsed_time
                            }
                            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 5)
                else:
                    connection_data['closed_eyes_start_time'] = None
                    # Solo resetear alertas de nivel 1 (las de nivel 2 requieren confirmación)
                    if connection_data.get('alert_level', 0) == 1:
                        connection_data['alert_level'] = 0

        # Codificar frame procesado con la información visual
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Si hay alerta, asegurarnos de que se envíe como JSON
        if alert_info:
            logger.info(f"Enviando alerta: {alert_info}")
            return buffer.tobytes(), alert_info

        # Si hay alerta activa, mantener el rectángulo visual
        if connection_data['critical_alert_active']:  # Cambiado de 'alert_active' a 'critical_alert_active'
            height, width = frame.shape[:2]
            if connection_data.get('alert_level', 0) == 2:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
            else:
                cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 5)

        return buffer.tobytes(), None

    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return frame_data, None

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("bytes") and len(message["bytes"]) == 1:
                command = message["bytes"][0]
                if command == 0:  # Reset completo
                    connection_data = manager.active_connections[websocket]
                    connection_data['closed_eyes_start_time'] = None
                    connection_data['alert_level'] = 0
                    connection_data['critical_alert_active'] = False
                    logger.info("Estado completamente reiniciado por señal del cliente")
                    
                    # Enviar confirmación de reinicio al cliente
                    reset_confirm = {
                        "type": "reset_confirm",
                        "message": "Estado reiniciado"
                    }
                    await websocket.send_text(json.dumps(reset_confirm))
                continue
            
            # Procesar frame normal
            if "bytes" in message:
                frame_data = message["bytes"]
                
                # Verificar tamaño máximo
                if len(frame_data) > MAX_FRAME_SIZE:
                    logger.warning("Frame demasiado grande recibido")
                    continue
                    
                # Procesar frame
                processed_frame, alert = await process_frame(
                    frame_data, 
                    manager.active_connections[websocket]
                )
                
                # Enviar frame procesado
                await websocket.send_bytes(processed_frame)
                
                # Enviar alerta si es necesario
                if alert:
                    alert_json = json.dumps(alert)
                    logger.info(f"Enviando alerta: {alert_json}")
                    await websocket.send_text(alert_json)
                
                # Actualizar timestamp
                manager.active_connections[websocket]['last_frame_time'] = time.time()
            
    except WebSocketDisconnect:
        logger.info("Cliente desconectado normalmente")
    except Exception as e:
        logger.error(f"Error en la conexión: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        manager.disconnect(websocket)