from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import cv2
import numpy as np
import base64
from eye_analyzer import EyeAnalyzer
from config import AlertConfig

# Inicializar FastAPI primero
app = FastAPI(
    title="Sistema de Detección de Somnolencia",
    description="API para la detección de somnolencia en tiempo real",
    version="1.0.0",
)

eye_analyzer = EyeAnalyzer()

class EyeStatus(BaseModel):
    status: str = Field(
        ..., 
        description="Estado actual del ojo",
        example="Open",
        enum=["Open", "Closed", "Unknown"]
    )

class FaceLocation(BaseModel):
    x: int = Field(..., description="Coordenada X del rostro", example=100)
    y: int = Field(..., description="Coordenada Y del rostro", example=100)
    width: int = Field(..., description="Ancho del área del rostro", example=200)
    height: int = Field(..., description="Alto del área del rostro", example=200)
    
    class Config:
        schema_extra = {
            "example": {
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 200
            }
        }

class FaceAnalysis(BaseModel):
    face_location: FaceLocation = Field(..., description="Ubicación del rostro en la imagen")
    left_eye_status: str = Field(..., description="Estado del ojo izquierdo")
    right_eye_status: str = Field(..., description="Estado del ojo derecho")
    is_active: bool = Field(..., description="Indica si la persona está activa o no")

class FrameData(BaseModel):
    frame: str = Field(
        ..., 
        description="Imagen codificada en base64",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    )

class DetectionResponse(BaseModel):
    status: str = Field(..., description="Estado de la respuesta")
    faces_detected: int = Field(..., description="Número de rostros detectados")
    results: List[FaceAnalysis] = Field(..., description="Resultados del análisis por cada rostro")

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz para verificar que la API está funcionando.
    """
    return {"message": "Sleep Detection API is running"}

@app.post(
    "/api/detect_drowsiness_live",
    response_model=DetectionResponse,
    summary="Detectar somnolencia en tiempo real",
    response_description="Resultados del análisis de somnolencia",
    tags=["Detección"]
)
async def detect_drowsiness_live(data: FrameData):
    """
    Analiza un frame de video para detectar signos de somnolencia.

    Parameters:
    - **data**: Frame en formato base64

    Returns:
    - **DetectionResponse**: Resultados del análisis
    """
    try:
        frame = _decode_frame(data.frame)
        if frame is None:
            raise HTTPException(
                status_code=400, 
                detail="No se pudo decodificar la imagen proporcionada"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = eye_analyzer.face_detector.detectMultiScale(
            gray,
            minNeighbors=5,
            scaleFactor=1.1,
            minSize=(25, 25)
        )

        results = [
            eye_analyzer.analyze_face(frame, gray, face_coords)
            for face_coords in faces
        ]

        return {
            'status': 'success',
            'faces_detected': len(results),
            'results': results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el procesamiento: {str(e)}"
        )

def _decode_frame(frame_data: str) -> np.ndarray:
    """
    Decodifica una imagen en base64 a un array de NumPy.

    Args:
        frame_data (str): String en base64 que contiene la imagen

    Returns:
        np.ndarray: Imagen decodificada en formato NumPy array

    Raises:
        ValueError: Si el formato de la imagen no es válido
    """
    try:
        # Eliminar el prefijo de data URL si existe
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("La imagen no pudo ser decodificada")
            
        return frame
    except Exception as e:
        raise ValueError(f"Error decodificando la imagen: {str(e)}")

# Configurar documentación OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Sistema de Detección de Somnolencia",
        version="1.0.0",
        description="""
        # Sistema de Detección de Somnolencia en Tiempo Real
        
        Esta API proporciona servicios para detectar somnolencia mediante análisis de video.
        
        ## Funcionalidades
        * Detección de estado de ojos
        * Análisis de somnolencia en tiempo real
        * Sistema de alertas
        * Procesamiento de video streaming
        """,
        routes=app.routes,
    )

    # Configurar tags
    openapi_schema["tags"] = [
        {
            "name": "General",
            "description": "Operaciones generales",
        },
        {
            "name": "Detección",
            "description": "Endpoints para detección de somnolencia",
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Asignar el schema personalizado
app.openapi = custom_openapi

# Configuración de Swagger UI
app.swagger_ui_parameters = {
    "defaultModelsExpandDepth": 1,
    "displayRequestDuration": True,
    "docExpansion": "list",
    "filter": True,
}
