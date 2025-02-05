import time
import logging
import json
from fastapi import WebSocket
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        """Inicializa el gestor de conexiones WebSocket."""
        self.active_connections: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """
        Acepta una nueva conexión WebSocket y la inicializa.
        
        Args:
            websocket (WebSocket): La conexión WebSocket a inicializar
        """
        await websocket.accept()
        self.active_connections[websocket] = {
            'closed_eyes_start_time': None,
            'last_frame_time': time.time(),
            'alert_level': 0,
            'critical_alert_active': False,
            'stream_active': False
        }
        logger.info("Nueva conexión establecida")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Cierra y elimina una conexión WebSocket.
        
        Args:
            websocket (WebSocket): La conexión WebSocket a cerrar
        """
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            logger.info("Conexión cerrada")

    async def send_alert(self, websocket: WebSocket, alert_info: dict) -> None:
        """
        Envía una alerta al cliente conectado.
        
        Args:
            websocket (WebSocket): La conexión WebSocket del cliente
            alert_info (dict): Información de la alerta a enviar
        """
        try:
            await websocket.send_text(json.dumps(alert_info))
            logger.info(f"Alerta enviada: {alert_info}")
        except Exception as e:
            logger.error(f"Error enviando alerta: {e}")

    async def reset_state(self, websocket: WebSocket) -> None:
        """
        Reinicia el estado de una conexión.
        
        Args:
            websocket (WebSocket): La conexión WebSocket a reiniciar
        """
        if websocket in self.active_connections:
            connection_data = self.active_connections[websocket]
            connection_data.update({
                'closed_eyes_start_time': None,
                'alert_level': 0,
                'critical_alert_active': False
            })
            await self.send_alert(websocket, {
                "type": "reset_confirm",
                "message": "Estado reiniciado"
            })
            logger.info("Estado de conexión reiniciado")

    def get_connection_state(self, websocket: WebSocket) -> Optional[dict]:
        """
        Obtiene el estado actual de una conexión.
        
        Args:
            websocket (WebSocket): La conexión WebSocket
            
        Returns:
            Optional[dict]: Estado de la conexión o None si no existe
        """
        return self.active_connections.get(websocket)

    def update_last_activity(self, websocket: WebSocket) -> None:
        """
        Actualiza el timestamp de última actividad de una conexión.
        
        Args:
            websocket (WebSocket): La conexión WebSocket a actualizar
        """
        if websocket in self.active_connections:
            self.active_connections[websocket]['last_frame_time'] = time.time()

    async def broadcast(self, message: str) -> None:
        """
        Envía un mensaje a todas las conexiones activas.
        
        Args:
            message (str): Mensaje a transmitir
        """
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error en broadcast a cliente: {e}")
                await self.disconnect(websocket)

    def is_connection_active(self, websocket: WebSocket) -> bool:
        """
        Verifica si una conexión está activa.
        
        Args:
            websocket (WebSocket): La conexión WebSocket a verificar
            
        Returns:
            bool: True si la conexión está activa, False en caso contrario
        """
        return websocket in self.active_connections
