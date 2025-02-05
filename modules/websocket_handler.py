import json
import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional
from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self, frame_processor: FrameProcessor):
        """
        Inicializa el manejador de WebSocket.
        
        Args:
            frame_processor (FrameProcessor): Procesador de frames
        """
        self.frame_processor = frame_processor
        self.MAX_FRAME_SIZE = 1024 * 1024
        self.DEFAULT_QUALITY = 80

    async def handle_message(self, message: dict, websocket: WebSocket, 
                           connection_data: dict) -> None:
        """
        Maneja los mensajes recibidos por el WebSocket.
        
        Args:
            message (dict): Mensaje recibido
            websocket (WebSocket): Conexión WebSocket
            connection_data (dict): Datos de la conexión
        """
        try:
            if self._is_control_message(message):
                await self._handle_control_message(message, websocket, connection_data)
                return

            if not self._is_valid_frame_message(message):
                logger.warning("Mensaje de frame inválido recibido")
                return

            await self._process_and_send_frame(message["bytes"], websocket, connection_data)

        except Exception as e:
            logger.error(f"Error manejando mensaje: {e}")
            await self._send_error(websocket, str(e))

    def _is_control_message(self, message: dict) -> bool:
        """Verifica si es un mensaje de control."""
        return (
            message.get("type") == "control" or
            (message.get("bytes") and len(message["bytes"]) == 1)
        )

    def _is_valid_frame_message(self, message: dict) -> bool:
        """Valida el mensaje de frame."""
        return (
            "bytes" in message and
            isinstance(message["bytes"], bytes) and
            len(message["bytes"]) <= self.MAX_FRAME_SIZE
        )

    async def _handle_control_message(self, message: dict, websocket: WebSocket, 
                                    connection_data: dict) -> None:
        """Maneja mensajes de control."""
        if message.get("bytes") == b'\x00':  # Reset command
            await self._handle_reset(websocket, connection_data)
        elif message.get("type") == "quality":
            self._handle_quality_change(message.get("value", self.DEFAULT_QUALITY))
        elif message.get("type") == "pause":
            await self._handle_pause(websocket, connection_data)
        elif message.get("type") == "resume":
            await self._handle_resume(websocket, connection_data)

    async def _handle_reset(self, websocket: WebSocket, connection_data: dict) -> None:
        """Maneja el comando de reset."""
        connection_data.update({
            'closed_eyes_start_time': None,
            'alert_level': 0,
            'critical_alert_active': False
        })
        await self._send_confirmation(websocket, "reset_confirm", "Estado reiniciado")

    def _handle_quality_change(self, quality: int) -> None:
        """Ajusta la calidad de compresión de imagen."""
        self.DEFAULT_QUALITY = max(1, min(100, quality))
        logger.info(f"Calidad de imagen ajustada a: {self.DEFAULT_QUALITY}")

    async def _handle_pause(self, websocket: WebSocket, connection_data: dict) -> None:
        """Pausa el procesamiento de frames."""
        connection_data['stream_active'] = False
        await self._send_confirmation(websocket, "pause_confirm", "Stream pausado")

    async def _handle_resume(self, websocket: WebSocket, connection_data: dict) -> None:
        """Reanuda el procesamiento de frames."""
        connection_data['stream_active'] = True
        await self._send_confirmation(websocket, "resume_confirm", "Stream reanudado")

    async def _process_and_send_frame(self, frame_data: bytes, websocket: WebSocket, 
                                    connection_data: dict) -> None:
        """Procesa y envía un frame."""
        if not connection_data.get('stream_active', True):
            return

        processed_frame, alert = await self.frame_processor.process_frame(
            frame_data, 
            connection_data
        )

        await websocket.send_bytes(processed_frame)
        
        if alert:
            await self._send_alert(websocket, alert)
        
        connection_data['last_frame_time'] = time.time()

    async def _send_confirmation(self, websocket: WebSocket, type_: str, message: str) -> None:
        """Envía una confirmación al cliente."""
        await websocket.send_text(json.dumps({
            "type": type_,
            "message": message
        }))

    async def _send_alert(self, websocket: WebSocket, alert: dict) -> None:
        """Envía una alerta al cliente."""
        await websocket.send_text(json.dumps(alert))
        logger.info(f"Alerta enviada: {alert}")

    async def _send_error(self, websocket: WebSocket, error: str) -> None:
        """Envía un mensaje de error al cliente."""
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": error
        }))
        logger.error(f"Error enviado al cliente: {error}")
