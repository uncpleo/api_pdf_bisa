from fastapi import FastAPI, File, UploadFile, HTTPException
import re
import joblib
import torch
import fitz  # PyMuPDF
import cv2
import easyocr
from transformers import BertTokenizer, BertModel
from collections import Counter
import os
from ultralytics import YOLO

# Configuración de la API
app = FastAPI(title="Procesador de PDF", description="API para extraer y clasificar información de PDFs")

# Cargar el tokenizer y el modelo BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
modelo = YOLO("modelos/best.pt")

# Cargar los modelos guardados
print("Cargando modelos...")
modelos_guardados = joblib.load('modelos/modelos_entrenados.pkl')
model_disciplina_robusto = modelos_guardados['model_disciplina_robusto']
model_clasificacion_robusto = modelos_guardados['model_clasificacion_robusto']
model_tipo_entregable = modelos_guardados['model_tipo_entregable']
nombres_entregables = modelos_guardados['nombres_entregables']
print("Modelos cargados exitosamente.")

# Lista de gerencias u oficinas válidas (incluyendo "00")
gerencias_validas = {
    "CEO", "SSMAC", "COM", "PCO", "DI", "GE", "MI", "MA", "GDP", "AD",
    "BBSS", "CONTR", "TIC", "TDC", "SSGG", "CONTA", "00"
}

# Expresión regular para detectar el código según la estructura dada
pattern = re.compile(r'\b([A-Z]{2,6}|00)(\d{3,4})O(\d{4})\b')

# Función para obtener embeddings de BERT
def get_bert_embeddings(texts, max_length=128):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Promedio de los embeddings
    return embeddings.numpy()

# Función para extraer el código más frecuente
def extract_most_frequent_code(text):
    matches = pattern.findall(text)  # Encuentra coincidencias con la expresión regular
    code_counts = Counter()  # Para contar cuántas veces aparece cada código
    first_appearance = {}  # Para almacenar el orden de aparición

    for gerencia, cliente, proyecto in matches:
        full_code = f"{gerencia}{cliente}O{proyecto}"

        # Verificar si la gerencia está en la lista de opciones válidas
        if gerencia in gerencias_validas:
            code_counts[full_code] += 1
            if full_code not in first_appearance:
                first_appearance[full_code] = len(first_appearance)  # Guarda el orden de aparición

    if not code_counts:
        return "No se encontró ningún código válido"

    # Ordenar por cantidad de repeticiones (descendente) y luego por orden de aparición
    most_frequent_code = sorted(code_counts.items(), key=lambda x: (-x[1], first_appearance[x[0]]))[0][0]

    return most_frequent_code

# Función para extraer el Código del Entregable
def extract_deliverable_code(text):
    pattern = r'\b([A-Z0-9]+(?:[-_][A-Z0-9]+)+)\b'
    matches = re.findall(pattern, text)
    if matches:
        return max(matches, key=len)
    return "No se encontró el código de entregable"

# Función principal para procesar el texto
def process_text(text):
    result = {
        'Código de proyecto': extract_most_frequent_code(text),
        'Disciplina': model_disciplina_robusto.predict(get_bert_embeddings([text]))[0],
        'Clasificación de entregable': model_clasificacion_robusto.predict(get_bert_embeddings([text]))[0],
        'Tipo de entregable': model_tipo_entregable.predict(get_bert_embeddings([text]))[0],
        'Código de entregable': extract_deliverable_code(text)
    }
    return result

# Función para convertir PDF a imágenes
def pdf_a_imagen(pdf_path, dpi=300):
    try:
        # Abrir el archivo PDF
        doc = fitz.open(pdf_path)
        
        # Convertir la primera página a imagen
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 180, dpi / 180)  # Escalamos la imagen
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Guardar la imagen temporalmente
        imagen_path = "temp_pagina.png"
        pix.save(imagen_path)
        return imagen_path
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return None

# Función para detectar ROI y extraer texto con EasyOCR
def detectar_roi_y_extraer_texto(imagen_path, modelo):
    try:
        # Cargar la imagen
        imagen = cv2.imread(imagen_path)
        
        # Predecir con el modelo YOLO
        resultados = modelo(imagen_path)
        
        # Procesar los resultados
        for resultado in resultados:
            cajas = resultado.boxes.xyxy  # Coordenadas de las cajas (x1, y1, x2, y2)
            for caja in cajas:
                x1, y1, x2, y2 = map(int, caja)  # Convertir coordenadas a enteros
                
                # Recortar la ROI
                roi = imagen[y1:y2, x1:x2]
                
                # Guardar temporalmente la ROI
                roi_path = "temp_roi.png"
                cv2.imwrite(roi_path, roi)
                
                # Aplicar OCR a la ROI con EasyOCR
                reader = easyocr.Reader(["es"])  # Especifica el idioma
                result = reader.readtext(roi_path)
                texto = " ".join([text for (bbox, text, prob) in result])
                
                # Eliminar la imagen temporal
                os.remove(roi_path)
                
                return texto  # Devolver el texto extraído
    except Exception as e:
        print(f"Error al procesar {imagen_path}: {e}")
        return None

# Función para extraer texto de la primera página
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        text = page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al extraer texto del PDF: {e}")

# Función principal para procesar el PDF
def process_pdf(pdf_path, modelo):
    # Abrir el PDF
    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    if num_pages > 1:
        # Si el PDF tiene más de una página, extraer texto de la primera página
        print("El PDF tiene más de una página. Extrayendo texto de la primera página...")
        text = extract_text_from_pdf(pdf_path)
    else:
        # Si el PDF tiene solo una página, convertirla a imagen
        print("El PDF tiene una sola página. Convirtiendo a imagen...")
        imagen_path = pdf_a_imagen(pdf_path)
        if imagen_path:
            # Encontrar la ROI y extraer texto
            print("Extrayendo texto de la ROI...")
            text = detectar_roi_y_extraer_texto(imagen_path, modelo)
            # Eliminar la imagen temporal
            os.remove(imagen_path)
        else:
            text = None

    if text:
        # Procesar el texto con los modelos
        result = process_text(text)
        print("\nResultados:")
        for key, value in result.items():
            print(f"{key}: {value}")
        return result  # Devolver el resultado
    else:
        print("No se pudo extraer texto del PDF.")
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

@app.post("/procesar-pdf/")
async def procesar_pdf(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        # Procesar el PDF y obtener el resultado
        result = process_pdf(temp_pdf_path, modelo)  # modelo_yolo debe ser cargado previamente

        # Eliminar el archivo temporal
        os.remove(temp_pdf_path)

        return result  # Devolver el resultado al cliente
    except Exception as e:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {e}")

# Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de procesamiento de PDFs"}

if __name__ == "__main__":  # Ejecutar la API
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)