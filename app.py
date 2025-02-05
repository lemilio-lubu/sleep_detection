from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import numpy as np
import base64
import cv2
from modules.face_detection import FaceDetector
from modules.eye_processor import EyeProcessor
from config import ModelPaths

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Sistema de Detección de Somnolencia",
        version="1.0.0",
        description="""
        # API de Detección de Somnolencia 👁️
        
        Esta API analiza imágenes en tiempo real para detectar signos de somnolencia.
        
        ## Características principales:
        
        * 🎯 Detección precisa de rostros
        * 👁️ Análisis del estado de los ojos
        * ⚡ Procesamiento en tiempo real
        * 🚨 Sistema de alertas por niveles
        
        ## Guía de uso:
        
        1. Capture un frame de video
        2. Codifique la imagen en base64
        3. Envíe al endpoint `/api/detect_drowsiness_live`
        4. Reciba el análisis de somnolencia
        
        ## Notas técnicas:
        
        * Formato de imagen: JPEG/PNG en base64
        * Resolución recomendada: 640x480
        * Iluminación: Ambiente bien iluminado
        """,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app = FastAPI(
    title="Sistema de Detección de Somnolencia",
    description="API para detección de somnolencia en tiempo real",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.openapi = custom_openapi

# Inicialización de componentes
eye_processor = EyeProcessor(
    model_path=ModelPaths.SLEEP_MODEL,
    left_cascade_path=ModelPaths.LEFT_EYE_CASCADE,
    right_cascade_path=ModelPaths.RIGHT_EYE_CASCADE
)

face_detector = FaceDetector(
    cascade_path=ModelPaths.FACE_CASCADE,
    eye_processor=eye_processor
)

class EyeStatus(BaseModel):
    status: str = Field(
        ...,
        description="Estado actual del ojo",
        example="Open",
        enum=["Open", "Closed", "Unknown"]
    )

class FaceLocation(BaseModel):
    x: int = Field(..., description="Coordenada X del rostro", example=100, ge=0)
    y: int = Field(..., description="Coordenada Y del rostro", example=100, ge=0)
    width: int = Field(..., description="Ancho del área del rostro", example=200, gt=0)
    height: int = Field(..., description="Alto del área del rostro", example=200, gt=0)

class FaceAnalysis(BaseModel):
    face_location: FaceLocation = Field(..., description="Ubicación del rostro detectado")
    left_eye_status: str = Field(..., description="Estado del ojo izquierdo", example="Open")
    right_eye_status: str = Field(..., description="Estado del ojo derecho", example="Open")
    is_active: bool = Field(..., description="Indica si la persona está activa", example=True)

class FrameData(BaseModel):
    frame: str = Field(
        ..., 
        description="Imagen codificada en base64",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    )

    @validator('frame')
    def validate_base64(cls, v):
        if not v.startswith(('data:image', 'iVBOR', '/9j/')):
            raise ValueError('Debe ser una imagen válida en formato base64')
        return v

class DetectionResponse(BaseModel):
    status: str = Field(..., description="Estado de la operación", example="success")
    faces_detected: int = Field(..., description="Número de rostros detectados", example=1, ge=0)
    results: List[FaceAnalysis] = Field(..., description="Resultados del análisis por cada rostro")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "faces_detected": 1,
                "results": [{
                    "face_location": {
                        "x": 100,
                        "y": 100,
                        "width": 200,
                        "height": 200
                    },
                    "left_eye_status": "Open",
                    "right_eye_status": "Open",
                    "is_active": True
                }]
            }
        }

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint de bienvenida y verificación de estado de la API.
    
    Returns:
        dict: Mensaje de bienvenida y estado
    """
    return {
        "status": "active",
        "message": "Bienvenido a la API de Detección de Somnolencia",
        "version": "1.0.0"
    }

@app.post(
    "/api/detect_drowsiness_live",
    response_model=DetectionResponse,
    tags=["Detección"],
    summary="Detectar somnolencia en tiempo real",
    response_description="Análisis de somnolencia del frame proporcionado",
    status_code=200,
    responses={
        200: {
            "description": "Análisis exitoso",
            "content": {
                "application/json": {
                    "example": DetectionResponse.Config.schema_extra["example"]
                }
            }
        },
        400: {
            "description": "Error en los datos de entrada",
            "content": {
                "application/json": {
                    "example": {"detail": "Imagen inválida o corrupta"}
                }
            }
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {"detail": "Error procesando la imagen"}
                }
            }
        }
    }
)
async def detect_drowsiness_live(data: FrameData):
    """
    Analiza un frame de video para detectar signos de somnolencia.
    
    Args:
        data (FrameData): Frame de video en formato base64
        
    Returns:
        DetectionResponse: Resultados del análisis de somnolencia
        
    Raises:
        HTTPException: Si hay errores en el procesamiento
        
    Ejemplo de uso:
    ```python
    import requests
    import base64
    
    # Leer imagen
    with open("frame.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    
    # Enviar request
    response = requests.post(
        "http://localhost:8000/api/detect_drowsiness_live",
        json={"frame": img_base64}
    )
    
    # Procesar resultados
    results = response.json()
    print(f"Rostros detectados: {results['faces_detected']}")
    ```
    """
    try:
        frame = decode_frame(data.frame)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        results = face_detector.detect_faces(frame)
        
        return {
            'status': 'success',
            'faces_detected': len(results),
            'results': results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def decode_frame(frame_data: str) -> np.ndarray:
    """Decodifica un frame desde base64."""
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Error decodificando frame: {str(e)}")
