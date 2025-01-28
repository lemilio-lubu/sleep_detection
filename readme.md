# Sistema de Detección de Somnolencia

Este sistema utiliza visión por computadora para detectar signos de somnolencia a través de la webcam, monitoreando el estado de los ojos del usuario.

## Requisitos Previos

- Python 3.8 o superior
- Webcam funcional
- Conexión a Internet (para la instalación inicial)

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/sleep_detection.git
cd sleep_detection
```

1. Crear un entorno virtual (recomendado):

```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En Linux/Mac
source venv/bin/activate
```

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Iniciar el servidor:

```bash
uvicorn modelo:app --reload
```

2. Acceder al cliente y ejecutar.

## Características

- Detección en tiempo real del estado de los ojos
- Alertas visuales y sonoras cuando se detecta somnolencia
- Interfaz web accesible desde cualquier navegador
- Sistema de alertas por niveles:
  - Nivel 1: Advertencia inicial (ojos cerrados por más de 2 segundos)
  - Nivel 2: Alerta crítica (ojos cerrados por más de 3 segundos)

## Notas de Uso

- Asegúrese de tener buena iluminación
- Mantenga una distancia apropiada de la cámara
- El sistema funciona mejor cuando se usa de frente a la cámara
- Para reiniciar las alertas, use el botón de reset en la interfaz

## Solución de Problemas

Si encuentra algún problema:

1. Verifique que la webcam esté correctamente conectada
2. Asegúrese de que ninguna otra aplicación esté usando la webcam
3. Reinicie el servidor si la conexión se vuelve inestable
4. Verifique que todos los requisitos estén instalados correctamente

## Licencia

MIT
