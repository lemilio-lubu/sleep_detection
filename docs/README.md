# Sistema de Detección de Somnolencia

## Descripción
Sistema de detección de somnolencia en tiempo real mediante análisis de video y machine learning.

## Características
- Detección de rostros en tiempo real
- Análisis del estado de los ojos
- Alertas de somnolencia
- API REST documentada
- WebSocket para streaming de video

## Requisitos
- Python 3.8+
- OpenCV
- TensorFlow
- FastAPI
- MediaPipe

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
1. Iniciar el servidor:
```bash
uvicorn app:app --reload
```

2. Acceder a la documentación:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Arquitectura
El sistema está compuesto por los siguientes módulos:

- `app.py`: API principal y endpoints REST
- `modelo.py`: Procesamiento de video y WebSocket
- `eye_analyzer.py`: Análisis de ojos y detección de somnolencia
- `config.py`: Configuración centralizada
- `connection_handler.py`: Gestión de conexiones WebSocket

## API
### REST Endpoints
- POST `/api/detect_drowsiness_live`: Analiza un frame de video

### WebSocket
- `/video_stream`: Streaming de video en tiempo real

## Contribuir
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar característica'`)
4. Push al branch (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## Licencia
MIT
