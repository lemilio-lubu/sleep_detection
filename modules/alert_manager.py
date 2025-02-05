from typing import Optional, Tuple, Dict
import time
import logging

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, 
                 alert_threshold: float = 2.0,
                 danger_threshold: float = 3.0,
                 eye_ar_threshold: float = 0.2):
        """
        Inicializa el gestor de alertas.

        Args:
            alert_threshold: Tiempo en segundos para la primera alerta
            danger_threshold: Tiempo en segundos para alerta crítica
            eye_ar_threshold: Umbral EAR para considerar ojos cerrados
        """
        self.ALERT_THRESHOLD = alert_threshold
        self.DANGER_THRESHOLD = danger_threshold
        self.EYE_AR_THRESH = eye_ar_threshold
        self.alert_levels = {
            1: "Advertencia: Signos de somnolencia",
            2: "¡PELIGRO! Somnolencia crítica detectada"
        }

    def check_alert_status(self, avg_ear: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """
        Determina el estado de alerta basado en EAR.
        
        Args:
            avg_ear: Valor promedio del Eye Aspect Ratio
            connection_data: Datos de la conexión actual
            
        Returns:
            Tuple[Optional[dict], bool]: (Información de alerta, Es alerta crítica)
        """
        if avg_ear >= self.EYE_AR_THRESH:
            self._reset_alert_status(connection_data)
            return None, False

        if connection_data['closed_eyes_start_time'] is None:
            connection_data['closed_eyes_start_time'] = time.time()

        elapsed_time = self._calculate_elapsed_time(connection_data)
        
        # Verificar alertas en orden de criticidad
        if elapsed_time > self.DANGER_THRESHOLD:
            return self._handle_danger_alert(elapsed_time, connection_data)
        
        if elapsed_time > self.ALERT_THRESHOLD:
            return self._handle_warning_alert(elapsed_time, connection_data)
        
        return None, False

    def _calculate_elapsed_time(self, connection_data: dict) -> float:
        """Calcula el tiempo transcurrido con ojos cerrados."""
        return time.time() - connection_data['closed_eyes_start_time']

    def _reset_alert_status(self, connection_data: dict) -> None:
        """
        Reinicia el estado de alerta.
        
        Args:
            connection_data: Datos de la conexión a reiniciar
        """
        connection_data['closed_eyes_start_time'] = None
        if connection_data.get('alert_level', 0) == 1:
            connection_data['alert_level'] = 0
            logger.info("Estado de alerta reiniciado")

    def _handle_danger_alert(self, elapsed_time: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """
        Maneja alertas de nivel crítico.
        
        Args:
            elapsed_time: Tiempo transcurrido con ojos cerrados
            connection_data: Datos de la conexión
            
        Returns:
            Tuple[Optional[dict], bool]: (Información de alerta crítica, True)
        """
        if connection_data['critical_alert_active']:
            return None, True

        alert_info = {
            "level": 2,
            "message": self.alert_levels[2],
            "elapsed_time": elapsed_time,
            "timestamp": time.time()
        }
        connection_data['critical_alert_active'] = True
        connection_data['alert_level'] = 2
        logger.warning(f"Alerta crítica activada: {elapsed_time:.1f}s")
        return alert_info, True

    def _handle_warning_alert(self, elapsed_time: float, connection_data: dict) -> Tuple[Optional[dict], bool]:
        """
        Maneja alertas de advertencia.
        
        Args:
            elapsed_time: Tiempo transcurrido con ojos cerrados
            connection_data: Datos de la conexión
            
        Returns:
            Tuple[Optional[dict], bool]: (Información de advertencia, True)
        """
        if connection_data['critical_alert_active']:
            return None, False

        alert_info = {
            "level": 1,
            "message": self.alert_levels[1],
            "elapsed_time": elapsed_time,
            "timestamp": time.time()
        }
        logger.info(f"Advertencia activada: {elapsed_time:.1f}s")
        return alert_info, True

    def update_thresholds(self, 
                         alert_threshold: Optional[float] = None,
                         danger_threshold: Optional[float] = None,
                         eye_ar_threshold: Optional[float] = None) -> None:
        """
        Actualiza los umbrales de alerta.
        
        Args:
            alert_threshold: Nuevo umbral para primera alerta
            danger_threshold: Nuevo umbral para alerta crítica
            eye_ar_threshold: Nuevo umbral EAR
        """
        if alert_threshold is not None:
            self.ALERT_THRESHOLD = max(0.5, alert_threshold)
        if danger_threshold is not None:
            self.DANGER_THRESHOLD = max(self.ALERT_THRESHOLD + 0.5, danger_threshold)
        if eye_ar_threshold is not None:
            self.EYE_AR_THRESH = max(0.1, min(0.4, eye_ar_threshold))
        
        logger.info(f"Umbrales actualizados: Alert={self.ALERT_THRESHOLD}, "
                   f"Danger={self.DANGER_THRESHOLD}, EAR={self.EYE_AR_THRESH}")
