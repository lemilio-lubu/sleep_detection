# procesar el video
# detectar el rostro
# analizamos el ojo derecho y el ojo izquierdo
# se predice si el ojo esta abierto o cerrado
# aanalizamos la somnolencia del conductor
# bajo la toma de decicion de somnolencia y sus ojos
# se envia la señal de alerta

import cv2
import numpy as np
import base64
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


face_detection = cv2.CascadeClassifier('haar\haarcascade_frontalface_alt.xml')
eyes_left_detection = cv2.CascadeClassifier('haar\haarcascade_lefteye_2splits.xml')
eyes_right_detection = cv2.CascadeClassifier('haar\haarcascade_righteye_2splits.xml')
captura = cv2.VideoCapture(0)

model = load_model('modelos\sleep_model_g.h5')

try:
    while True:
        ret, frame = captura.read()
        alto, ancho = frame.shape[:2]

        # si no captura un frame se rompe el ciclo
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(
            gray,
            minNeighbors=5,
            scaleFactor=1.1,
            minSize=(25, 25)
        )
        left_eye = eyes_left_detection.detectMultiScale(gray)
        right_eye = eyes_right_detection.detectMultiScale(gray)

        
        # dibuja un rectangulo en la cara
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #    roi_color = frame[y:y+h, x:x+w]
        #    face_img = cv2.resize(frame, (64, 64))
        #    face_img = face_img.astype('float')/255.0
        #    face_img = np.expand_dims(face_img, axis=0)

        #    prediction = model.predict(face_img)
        #    yawn_prob = np.argmax(prediction, axis=1)

        #    print(prediction)

        
        
        
        # dibujar rectangulo en el ojo derecho
        for (x, y, w, h) in right_eye:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_color = frame[y:y+h, x:x+w]
            eye_img = cv2.resize(roi_color, (64, 64))
            eye_img = eye_img.astype('float')/255.0
            eye_img = np.expand_dims(eye_img, axis=0)

            prediction = model.predict(eye_img)
            prob = prediction[0]
            eye_claa = np.argmax(prob)

            if eye_claa == 0:
                status = 'Open'
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif eye_claa == 1:
                status = 'Closed'  
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print(eye_claa)
        # dibujar rectangulo en el ojo izquierdo
        #for (x, y, w, h) in left_eye:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # lo plasma en la pantalla
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    captura.release()
    cv2.destroyAllWindows()
