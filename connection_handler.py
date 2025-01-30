from fastapi import WebSocket
from typing import Dict, Optional
import time
import logging
import json

class ConnectionHandler:
    def __init__(self):
        self.active_connections: Dict[WebSocket, dict] = {}
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket) -> None:
        """Initialize new WebSocket connection."""
        await websocket.accept()
        self.active_connections[websocket] = {
            'closed_eyes_start_time': None,
            'last_frame_time': time.time(),
            'alert_level': 0,
            'critical_alert_active': False
        }
        self.logger.info("Nueva conexión establecida")

    def disconnect(self, websocket: WebSocket) -> None:
        """Clean up disconnected WebSocket."""
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            self.logger.info("Conexión cerrada")

    async def reset_connection_state(self, websocket: WebSocket) -> None:
        """Reset connection state and send confirmation."""
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
