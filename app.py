from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Inicializar la aplicación
app = FastAPI()

# Cargar los clasificadores Haar Cascade
face_detection = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
left_eye_detection = cv2.CascadeClassifier('haar/haarcascade_lefteye_2splits.xml')
right_eye_detection = cv2.CascadeClassifier('haar/haarcascade_righteye_2splits.xml')

# Cargar el modelo de ML
model = load_model('modelos/sleep_model.h5')

# Modelo de entrada para FastAPI
class FrameData(BaseModel):
    frame: str  # Base64 string del fotograma

def process_eye(eye_frame):
    """Procesa la imagen del ojo y hace la predicción"""
    eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    eye = eye / 255.0
    eye = eye.reshape(24, 24, -1)
    eye = np.expand_dims(eye, axis=0)
    prediction = model.predict(eye)
    return np.argmax(prediction, axis=1)[0]

@app.post("/api/detect_drowsiness_live")
async def detect_drowsiness_live(data: FrameData):
    try:
        # Decodificar el fotograma
        frame_bytes = base64.b64decode(data.frame)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_detection.detectMultiScale(
            gray,
            minNeighbors=5,
            scaleFactor=1.1,
            minSize=(25, 25)
        )

        results = []
        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]

            # Detectar ojos en la región del rostro
            left_eyes = left_eye_detection.detectMultiScale(face_gray)
            right_eyes = right_eye_detection.detectMultiScale(face_gray)

            # Variables para almacenar predicciones
            left_eye_status = "Unknown"
            right_eye_status = "Unknown"
            is_active = True

            # Procesar ojo derecho
            for (ex, ey, ew, eh) in right_eyes:
                right_eye_frame = face_frame[ey:ey+eh, ex:ex+ew]
                right_eye_pred = process_eye(right_eye_frame)
                right_eye_status = "Open" if right_eye_pred == 1 else "Closed"
                break

            # Procesar ojo izquierdo
            for (ex, ey, ew, eh) in left_eyes:
                left_eye_frame = face_frame[ey:ey+eh, ex:ex+ew]
                left_eye_pred = process_eye(left_eye_frame)
                left_eye_status = "Open" if left_eye_pred == 1 else "Closed"
                break

            # Determinar si está activo o inactivo
            if left_eye_status == "Closed" and right_eye_status == "Closed":
                is_active = False

            results.append({
                'face_location': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'left_eye_status': left_eye_status,
                'right_eye_status': right_eye_status,
                'is_active': is_active
            })

        return {
            'status': 'success',
            'faces_detected': len(results),
            'results': results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
