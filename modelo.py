import logging
import traceback
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from modules.connection_manager import ConnectionManager
from modules.frame_processor import FrameProcessor
from modules.websocket_handler import WebSocketHandler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar la aplicación FastAPI
app = FastAPI()

# Configuración de MediaPipe
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

# Inicializar componentes
manager = ConnectionManager()
frame_processor = FrameProcessor(face_mesh, LEFT_EYE, RIGHT_EYE)
websocket_handler = WebSocketHandler(frame_processor)

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
        logger.error(traceback.format_exc())
    finally:
        manager.disconnect(websocket)