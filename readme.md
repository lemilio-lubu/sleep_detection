# 🚀 Sistema de Detección de Somnolencia 👀

## 📝 Descripción
Sistema de detección de somnolencia en tiempo real mediante análisis de video y machine learning. ¡Mantente alerta y seguro! 🚨

---

## ✨ Características
- 👤 **Detección de rostros en tiempo real**
- 👁️ **Análisis del estado de los ojos**
- ⚠️ **Alertas de somnolencia**
- 📄 **API REST documentada**
- 📡 **WebSocket para streaming de video**

---

## 📋 Requisitos
Asegúrate de tener instalados los siguientes paquetes y dependencias:

- 🐍 **Python 3.8+**
- 🖼️ **OpenCV**
- 🧠 **TensorFlow**
- 🚀 **FastAPI**
- ✋ **MediaPipe**

---

## 🛠️ Instalación
Ejecuta el siguiente comando para instalar todas las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## 🚦 Uso
Para iniciar el servidor, ejecuta:

```bash
uvicorn app:app --reload
```

Accede a la documentación interactiva:
- 📚 **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📖 **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🏗️ Arquitectura
El sistema está compuesto por los siguientes módulos:

- **`app.py`**: API principal y endpoints REST 🛠️
- **`modelo.py`**: Procesamiento de video y WebSocket 🎥
- **`eye_analyzer.py`**: Análisis de ojos y detección de somnolencia 👁️
- **`config.py`**: Configuración centralizada ⚙️
- **`connection_handler.py`**: Gestión de conexiones WebSocket 📡

---

## 🌐 API

### REST Endpoints
- **POST** `/api/detect_drowsiness_live`: Analiza un frame de video 🖼️

### WebSocket
- `/video_stream`: Streaming de video en tiempo real 📹

---

## 🤝 Contribuir
¡Tu contribución es bienvenida! Sigue estos pasos:

1. 🍴 **Haz un fork** del repositorio
2. 🌿 **Crea una rama feature**
   ```bash
   git checkout -b feature/nueva-caracteristica
   ```
3. 💾 **Haz commit de tus cambios**
   ```bash
   git commit -am 'Agregar nueva característica'
   ```
4. 🚀 **Sube la rama**
   ```bash
   git push origin feature/nueva-caracteristica
   ```
5. 📤 **Crea un Pull Request**

---

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**.

---

© 2025 - Sistema de Detección de Somnolencia

