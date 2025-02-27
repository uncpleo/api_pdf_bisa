# API de Extracción de Información de PDFs
Esta API permite extraer y clasificar información de archivos PDF, como códigos de proyecto, disciplinas, clasificaciones de entregables y más. Utiliza modelos de machine learning y técnicas de procesamiento de texto e imágenes para realizar estas tareas.

## Estructura del Proyecto
├── __pycache__
│  └── rxconfig.cpython-39.pyc
├── .web
│  ├── components
│  ├── public
│  └── utils
├── api_extraccion
│  ├── __init__.py
│  ├── modelos
│  │   ├── best.pt          # Modelo YOLO para detección de ROI
│  │   ├── modelos_entrenados.pkl  # Modelos de ML entrenados
│  └── api_extraccion.py    # Código principal de la API
├── assets
│  └── favicon.ico
├── .gitignore
├── requeriments.txt        # Dependencias del proyecto
└── rxconfig.py

## Requisitos Previos
Python 3.9 o superior: Asegúrate de tener Python instalado. Puedes verificarlo con:


## Instalación de Dependencias:
Instala las dependencias necesarias usando el archivo requirements.txt:

pip install -r requirements.txt


## Instrucciones para Ejecutar la API

cd api_extraccion
python api_extraccion.py


La API estará disponible en http://127.0.0.1:8000. Puedes probarla usando:
Endpoint raíz: GET / (Mensaje de bienvenida).
Procesar PDF: POST /procesar-pdf/ (Sube un archivo PDF para extraer información).

## Dependencias (requirements.txt)
Asegúrate de que tu archivo requirements.txt contenga las siguientes dependencias:

fastapi==0.95.2
uvicorn==0.22.0
torch==2.0.1
transformers==4.30.2
joblib==1.2.0
pymupdf==1.22.5
opencv-python==4.7.0.72
easyocr==1.6.2
numpy==1.24.3
scikit-learn==1.2.2
python-multipart==0.0.6

## Para generar o actualizar el archivo requirements.txt, puedes usar:

pip freeze > requirements.txt

## Notas Adicionales

Modelo YOLO:
El archivo best.pt en la carpeta modelos es el modelo YOLO preentrenado para detectar regiones de interés (ROI) en las imágenes. Asegúrate de que esté correctamente ubicado.

Modelos de Machine Learning:
Los modelos de clasificación (modelos_entrenados.pkl) deben estar en la carpeta modelos. Estos modelos se cargan automáticamente al iniciar la API.

PDFs con Múltiples Páginas:
Si el PDF tiene más de una página, la API solo procesará la primera página. Para procesar páginas adicionales, puedes modificar el código en api_extraccion.py.

PDFs con Imágenes:
La API convierte la primera página del PDF en una imagen si el PDF tiene una sola página. Luego, utiliza EasyOCR para extraer texto de las regiones de interés detectadas por YOLO.
