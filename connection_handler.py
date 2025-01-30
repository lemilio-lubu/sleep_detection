from fastapi import WebSocket
from typing import Dict, Optional
import time
import logging
import json
import asyncio
from fastapi import WebSocketDisconnect

class ConnectionHandler:
    def __init__(self):
        self.active_connections: Dict[WebSocket, dict] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[websocket] = {
            'closed_eyes_start_time': None,
            'last_frame_time': time.time(),
            'alert_level': 0,
            'critical_alert_active': False,
            'stream_active': False
        }
        self.logger.info("Nueva conexión establecida")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            self.logger.info("Conexión cerrada")

    async def reset_connection_state(self, websocket: WebSocket) -> None:
        connection_data = self.active_connections[websocket]
        connection_data.update({
            'closed_eyes_start_time': None,
            'alert_level': 0,
            'critical_alert_active': False
        })
        await websocket.send_text(json.dumps({
            "type": "reset_confirm",
            "message": "Estado reiniciado"
        }))
        self.logger.info("Estado reiniciado por señal del cliente")

    # En el método handle_websocket_messages
    async def handle_websocket_messages(self, websocket: WebSocket):
        try:
            while True:
                message = await websocket.receive_text()
                self.logger.info(f"Mensaje recibido: {message}")  # <-- Nuevo log
                if message == "start_stream":
                    await self.handle_start_stream(websocket)
                # ...
                elif message == "reset_alert":
                    await self.reset_connection_state(websocket)
        except WebSocketDisconnect:
            self.logger.info("Cliente desconectado")
        except Exception as e:
            self.logger.error(f"Error en la conexión: {str(e)}")
        finally:
            self.disconnect(websocket)

    async def handle_start_stream(self, websocket: WebSocket):
        connection_data = self.active_connections.get(websocket)
        if connection_data:
            connection_data['stream_active'] = True
            self.logger.info("Iniciando transmisión de video")
            # Iniciar envío de frames en segundo plano
            asyncio.create_task(self.send_video_frames(websocket))

    # En el método send_video_frames
    async def send_video_frames(self, websocket: WebSocket):
        connection_data = self.active_connections.get(websocket)
        while connection_data and connection_data.get('stream_active'):
            try:
                self.logger.info("Enviando frame...")  # <-- Nuevo log
                # ... (tu lógica de envío de frames)
                fake_frame = b"\x00" * 1024  # Frame de prueba de 1KB
                await websocket.send_bytes(fake_frame)
                await asyncio.sleep(0.033)  # ~30 FPS
            except Exception as e:
                self.logger.error(f"Error enviando frame: {str(e)}")
                break