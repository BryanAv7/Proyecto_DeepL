# Librerias por utilizar
# verison de Script: v3.2
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd  
import google.generativeai as genai
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import base64
import difflib
from rapidfuzz import fuzz
from sentence_transformers.util import cos_sim


from google.cloud import vision
from google.cloud.vision_v1 import types
import os
import io
import re

from ultralytics import YOLO  
 

app = FastAPI()

# Configuracion RUTAS
RUTA_DATASET = "../modelos/data_actualizadoArreglado.csv"
RUTAS_MODELO_YOLO = "../modelos/last.pt"
RUTAS_MODELO_CNN = "../modelos/modelo_productosbeta.pth"
RUTAS_OCR = "../modelos/ocr-productos-466521-3734397fcb54.json" 
print("Backend > Rutas definidas")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir la arquitectura SimpleCNN
# Definir la arquitectura RobustCNN (modelo correcto)
class RobustCNN(nn.Module):
    def __init__(self, num_classes):
        super(RobustCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32x32 -> 16x16

        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Variables globales para los modelos
yolo_model = None
cnn_model = None
cnn_transform = None
id2label = None
productos_df = None  # ✅ DataFrame con los datos de productos

#-----------------------------------------------------------
# Funciones ASISTENTE INTELIGENTE

modelo = SentenceTransformer("all-mpnet-base-v2")

COLUMNAS_EMB = [
    "Nombre", "Descripcion", "Categoria", "Tipo_promocion_aplicable",
    "Nivel_ventas", "Caducidad_aproximada", "Nivel_stock", "Rango_precio"
]

def aplicar_ingenieria_variables(df):
    hoy = pd.Timestamp(datetime.today().date())

    df["Nivel_ventas"] = pd.cut(
        df["Historico_ventas_semanales"],
        bins=[-1, 50, 80, float("inf")],
        labels=["Baja demanda", "Demanda media", "Alta demanda"]
    )

    df["Dias_para_caducar"] = (pd.to_datetime(df["Fecha_Caducidad"]) - hoy).dt.days
    df["Caducidad_aproximada"] = pd.cut(
        df["Dias_para_caducar"],
        bins=[-1, 7, 30, 90, float("inf")],
        labels=["por caducar", "Próxima a caducar", "Caduca en mediano plazo", "Caduca a largo plazo"]
    )

    df["Nivel_stock"] = pd.cut(
        df["Stock"],
        bins=[-1, 30, 80, float("inf")],
        labels=["Stock bajo", "Stock medio", "Stock alto"]
    )

    df["Rango_precio"] = pd.cut(
        df["Costo"],
        bins=[-1, 0.85, 1.2, 1.9, float("inf")],
        labels=["Muy económico", "Económico", "Costo medio", "Costoso"]
    )

    return df

def texto_estructurado(row, pregunta=""):
    base = (
        f"Nombre: {row['Nombre']}. "
        f"Descripcion: {row['Descripcion']}. "
        f"Categoria: {row['Categoria']}. "
        f"Tipo de Promoción: {row['Tipo_promocion_aplicable']}. "
        f"Nivel de ventas: {row['Nivel_ventas']}. "
        f"Estado de caducidad: {row['Caducidad_aproximada']}. "
        f"Nivel de stock: {row['Nivel_stock']}. "
        f"Rango de precio: {row['Rango_precio']}."
    )

    refuerzo = " " + str(row['Categoria']) * 2
    if "caduc" in pregunta.lower():
        refuerzo += " " + str(row['Caducidad_aproximada']) * 3

    return base + refuerzo

def cargar_inventario():
    df = pd.read_csv(RUTA_DATASET)
    df = aplicar_ingenieria_variables(df)

    # Generar texto enriquecido con estructura y refuerzo
    df["texto_busqueda"] = df.apply(lambda row: texto_estructurado(row, ""), axis=1)
    df["embedding"] = df["texto_busqueda"].apply(lambda x: modelo.encode(x))
    return df

inventario_df = cargar_inventario()

# Funciones Integracion LLM -------

def construir_prompt_llm(pregunta: str, resultados: list) -> str:
    prompt = f"""Eres un asistente inteligente para un minimarket. Un usuario ha preguntado: "{pregunta}"

Basándote en los siguientes productos encontrados, genera una respuesta amigable y útil para el cliente:

"""
    for idx, item in enumerate(resultados, 1):
        prompt += f"""{idx}. {item["Nombre"]} - {item["Descripcion"]} | Categoria: {item["Categoria"]}, Costo: ${item["Costo"]}, Stock: {item["Stock"]}, Fecha_Caducidad: {item["Fecha_Caducidad"]}\n"""

    prompt += "\nResponde de forma clara y útil:"
    return prompt

genai.configure(api_key="...")  # ← Reemplaza con tu clave temporal

#modelo_gemini = genai.GenerativeModel("gemini-pro")
modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")

def obtener_respuesta_llm(prompt: str) -> str:
    response = modelo_gemini.generate_content(prompt)
    return response.text.strip()

#-----------------------------------------------------------
# END-POINTS ⭕

@app.on_event("startup")
async def load_models():
    global yolo_model, cnn_model, cnn_transform, id2label, productos_df

    try:
        # Cargar modelo YOLOv8
        yolo_model = YOLO(RUTAS_MODELO_YOLO)  
        print("✅ Modelo YOLOv8 cargado correctamente")
        
        # Definir el mapeo de etiquetas 
        id2label = {
            '79ae74fa': 0,
            '642fcf3e': 1,
            '90a8f2a5': 2,
            '274ed5aa': 3,
            'd8e209f7': 4,
            'a95ab3e9': 5,
            'af8afcd9': 6,
            'ae59bd37': 7,
            '8dad85bf': 8,
            '7f7d3c1f': 9,
            '2faed7a6': 10,
            'ce2fe367': 11,
            '7dc3af65': 12,
            'a6decf91': 13,
            '86f78320': 14
        }

        num_classes = len(id2label)

        # Instanciar y cargar modelo CNN
        cnn_model = RobustCNN(num_classes=num_classes)
        checkpoint = torch.load(RUTAS_MODELO_CNN, map_location=torch.device('cpu'))
        cnn_model.load_state_dict(checkpoint['modelo_estado'])
        cnn_model.eval()
        print("✅ Modelo CNN cargado correctamente")

        # Transformaciones para el modelo CNN (ajustadas a 128x128)
        cnn_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ✅ Cargar archivo CSV con información detallada de productos
        productos_df = pd.read_csv(RUTA_DATASET)
        print("✅ Datos de productos cargados correctamente")

    except Exception as e:
        print(f"❌ Error al cargar los modelos o CSV: {str(e)}")
        raise RuntimeError(f"Error al cargar los modelos: {str(e)}")

# Nueva función multilabel para detecciones YOLO (IDs + confianza + coordenadas)
async def get_all_detected_labels(image_bytes, conf_threshold=0.20):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = yolo_model(img_cv, conf=conf_threshold)
        detections = results[0].boxes

        if len(detections) == 0:
            print("⚠️ YOLO no detectó ningún objeto con la confianza mínima.")
            return []

        idx2id = {v: k for k, v in id2label.items()}
        productos_detectados = []

        # ✅ Filtrar una sola detección por clase (la de mayor confianza)
        detecciones_por_clase = {}
        for box in detections:
            class_idx = int(box.cls.item())
            if class_idx not in detecciones_por_clase:
                detecciones_por_clase[class_idx] = box
            else:
                if box.conf.item() > detecciones_por_clase[class_idx].conf.item():
                    detecciones_por_clase[class_idx] = box

        for class_idx, box in detecciones_por_clase.items():
            confidence = float(box.conf.item())
            box_coords = list(map(int, box.xyxy[0].tolist()))
            label_str = idx2id.get(class_idx, str(class_idx))

            productos_detectados.append({
                "id": label_str,
                "confidence": confidence,
                "box": box_coords
            })

        print(f"🔎 Productos detectados: {productos_detectados}")
        return productos_detectados

    except Exception as e:
        print(f"❌ Error en get_all_detected_labels: {str(e)}")
        raise RuntimeError(f"Error en get_all_detected_labels: {str(e)}")


# Endpoint adicional para obtener las detecciones YOLO múltiples
@app.post("/api/yolo-detections")
async def detectar_con_yolo(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        image_bytes = await file.read()
        detecciones = await get_all_detected_labels(image_bytes)
        return {"detecciones": detecciones}

    except Exception as e:
        print(f"❌ Error en /api/yolo-detections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mantener el flujo original para una sola detección
async def get_most_confident_detection(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = yolo_model(img_cv, conf=0.20)
        detections = results[0].boxes  

        if len(detections) == 0:
            print("⚠️ YOLO no detectó ningún objeto con la confianza mínima.")
            return None, None

        best_box = max(detections, key=lambda b: b.conf)
        print(f"🎯 Mejor detección con confianza: {best_box.conf.item():.4f}")

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        crop = img_cv[y1:y2, x1:x2]
        if crop.size == 0:
            print("⚠️ La región recortada está vacía.")
            return None, None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), None

    except Exception as e:
        print(f"❌ Error en procesamiento YOLO: {str(e)}")
        raise RuntimeError(f"Error en procesamiento YOLO: {str(e)}")


    except Exception as e:
        print(f"❌ Error en procesamiento YOLO: {str(e)}")
        raise RuntimeError(f"Error en procesamiento YOLO: {str(e)}")
    

# Función para realizar la predicción OCR
def prediccion_ocr(crop_rgb):

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = RUTAS_OCR

    pil_img = Image.fromarray(crop_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    content = buf.getvalue()

    client = vision.ImageAnnotatorClient()
    image = types.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        print(f"❌ Error OCR: {response.error.message}")
        return None

    texts = response.text_annotations
    if texts:
        texto_completo = texts[0].description.strip()
        lineas = texto_completo.splitlines()

        mejor_linea = ""
        mejor_limpio = ""

        for linea in lineas:
            limpia = re.sub(r'[^a-zA-Z]', '', linea).strip()
            if len(limpia) >= 4:
                mejor_linea = linea.strip()
                mejor_limpio = limpia
                break  

            if not mejor_limpio and len(limpia) > 0:
                mejor_linea = linea.strip()
                mejor_limpio = limpia

        if mejor_limpio:
            print("\n::::::::::::::::::::..::::::::::::::::::::\n")
            print(f"🔍Texto extraído por el OCR: {mejor_linea}")
            return mejor_linea
        else:
            print("⚠️ OCR no extrajo texto útil.")
            return None

    else:
        print("No se detectó texto.")
        return None

# Función para predecir el ID del producto usando CNN
async def predict_product_id(image_crop):
    try:
        img_pil = Image.fromarray(image_crop)
        input_tensor = cnn_transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_id = pred.item()

        label = next((k for k, v in id2label.items() if v == pred_id), str(pred_id))
        print(f"🔍 Predicción CNN: Clase {pred_id} → Etiqueta {label}")

        return label

    except Exception as e:
        print(f"❌ Error en predicción CNN: {str(e)}")
        raise RuntimeError(f"Error en predicción CNN: {str(e)}")
    
    
# Función para comparar OCR con CNN
def coincidencia_ocr_cnn(texto_ocr: str, producto_cnn: dict, umbral_similitud: float = 30.0) -> str:
    import difflib

    if not texto_ocr or "Nombre" not in producto_cnn:
        return "❌ Producto No Coincidente"

    nombre_producto = str(producto_cnn["Nombre"]).lower()
    texto_ocr = texto_ocr.lower()

    similitud = difflib.SequenceMatcher(None, texto_ocr, nombre_producto).ratio() * 100
    print("\n::::::::::::::::::::..::::::::::::::::::::\n")
    print("🔎 Comparando OCR con CNN:")
    print(f"OCR     → {texto_ocr}")
    print(f"CNN     → {nombre_producto}")
    print(f"📊 Similitud: {similitud:.2f}%")
    
    if similitud >= umbral_similitud:
        return "✅ Producto Coincidente"
    
    else:
        return "❌ Producto No Coincidente"


# Función de búsqueda de coincidencias con respaldo fuzzy y embeddings
def busqueda_comparacion(texto_ocr: str, df: pd.DataFrame, modelo_embed, umbral=0.30):
    if not texto_ocr:
        print("⚠️ Texto OCR vacío. No se puede buscar.")
        return None

    texto_ocr_limpio = texto_ocr.lower().strip()
    print("\n:::::::::::::::Opción de Respaldo::::::::::::::::\n")
    print(f"🔍 Buscando coincidencia para OCR: '{texto_ocr_limpio}'")

    # Si la palabra capturada por OCR es Corta(menos de 3 letras), se aplica búsqueda fuzzy
    if len(texto_ocr_limpio) <= 3:
        print("🔎 Palabra OCR es corta, aplicando búsqueda fuzzy...")
        for _, row in df.iterrows():
            nombre = str(row['Nombre']).lower().strip()
            score = fuzz.partial_ratio(texto_ocr_limpio, nombre) / 100.0
            if score >= 0.80:
                print(f"📌 Producto sugerido (por fuzzy): {row['Nombre']}")
                print(f"🔍 Resultado final: {row['Nombre']} con score {score*100:.2f}%")
                return row.to_dict()
        print("⚠️ No se encontró coincidencia fuzzy suficiente. Continuando con embeddings...\n")

    # Si la palabra capturada por OCR No es corta(menos de 3 letras), se usa normalmente embeddings
    emb_ocr = modelo_embed.encode(texto_ocr_limpio)

    for _, row in df.iterrows():
        nombre = str(row['Nombre']).lower().strip()
        descripcion = str(row['Descripcion']).lower().strip()

        emb_nombre = modelo_embed.encode(nombre)
        emb_descripcion = modelo_embed.encode(descripcion)

        sim_nombre = cos_sim(emb_ocr, emb_nombre).item()
        sim_desc = cos_sim(emb_ocr, emb_descripcion).item()

        mejor_sim = max(sim_nombre, sim_desc)

        if mejor_sim >= umbral:
            print(f"🔍 Resultado final: {row['Nombre']} con similitud {mejor_sim*100:.2f}%")
            return row.to_dict()

    print("❌ No se encontró ningún producto con similitud suficiente.")
    return None


@app.post("/api/predict-product")
async def predict_product(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer imagen
        image_bytes = await file.read()

        # Convertir imagen a OpenCV
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ✅ Obtener solo una detección por clase (la de mayor confianza)
        detecciones_filtradas = await get_all_detected_labels(image_bytes)

        if len(detecciones_filtradas) == 0:
            raise HTTPException(status_code=400, detail="No se detectaron productos en la imagen")

        productos_detectados = []

        for det in detecciones_filtradas:
            x1, y1, x2, y2 = det["box"]
            crop = img_cv[y1:y2, x1:x2]

            if crop.size == 0:
                print("⚠️ Detección ignorada por recorte vacío.")
                continue

            # Convertir recorte a RGB y ejecutar OCR
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            texto_ocr = prediccion_ocr(crop_rgb)  # ← Solo consola

            # Codificar imagen recortada como base64
            _, buffer = cv2.imencode('.jpg', crop_rgb)
            imagen_crop_base64 = base64.b64encode(buffer).decode('utf-8')

            # Clasificación CNN
            etiqueta_cnn = await predict_product_id(crop_rgb)

            # Buscar información del producto por ID CNN
            producto_info = productos_df[productos_df['ID'] == etiqueta_cnn]

            if not producto_info.empty:
                producto_dict = producto_info.iloc[0].to_dict()
                producto_dict["imagen_crop_base64"] = imagen_crop_base64
                productos_detectados.append(producto_dict)

                # Comparar OCR vs CNN
                resultado_validacion = coincidencia_ocr_cnn(texto_ocr, producto_dict)
                print(f"➳Resultado de comparación: {resultado_validacion}")

                # Si no hay coincidencia, buscar sugerencia (solo consola) y usar producto sugerido
                if "No Coincidente" in resultado_validacion:
                    sugerido = busqueda_comparacion(texto_ocr, productos_df, modelo)
                    if sugerido:
                        print(f"📌 Producto sugerido: {sugerido['Nombre']}")

                        # Buscar info del producto sugerido por ID y reemplazar producto_dict
                        producto_info_sug = productos_df[productos_df['ID'] == sugerido['ID']]
                        if not producto_info_sug.empty:
                            producto_dict_sug = producto_info_sug.iloc[0].to_dict()
                            producto_dict_sug["imagen_crop_base64"] = imagen_crop_base64

                            # Reemplazar el último producto agregado con el sugerido
                            productos_detectados[-1] = producto_dict_sug
                    else:
                        print("📌 Sin sugerencia encontrada.")
            else:
                print(f"⚠️ Producto con ID {etiqueta_cnn} no encontrado en CSV.")

        if not productos_detectados:
            raise HTTPException(status_code=404, detail="No se encontró información de los productos detectados.")

        # ✅ Devolver la lista de productos clasificados
        return {"productos": productos_detectados}

    except Exception as e:
        print(f"❌ Error general en predict-product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------------
# Endpoints adicionales para el asistente inteligente
@app.get("/api/saludo")
def obtener_saludo():
    return {"mensaje": "🚀¡Aplicación de Inteligencia Artificial para el Reconocimiento de Productos!"}


@app.post("/api/asistente")
async def asistente(request: Request):
    global inventario_df
    data = await request.json()
    pregunta = data["mensaje"]

    # Recargar CSV actualizado y aplicar ingeniería
    inventario_df = cargar_inventario()

    emb_pregunta = modelo.encode(pregunta)
    inventario_df["score"] = inventario_df["embedding"].apply(
        lambda x: util.cos_sim(emb_pregunta, x).item()
    )

    # Dar peso extra a productos "por caducar"
    peso_caducar = 0.15  # Ajusta este valor para la importancia deseada
    inventario_df.loc[
        inventario_df["Caducidad_aproximada"] == "por caducar", "score"
    ] += peso_caducar

    score_mean = inventario_df["score"].mean()
    score_std = inventario_df["score"].std()
    score_max = inventario_df["score"].max()
    umbral_final = max(score_mean + 0.1 * score_std, score_max * 0.85)

    resultados_filtrados = inventario_df[inventario_df["score"] >= umbral_final]
    top_n = resultados_filtrados.sort_values("score", ascending=False).head(3)

    print("\n=== 🔎 Similitud ===")
    print(f"📝 Pregunta: {pregunta}")
    print(f"📊 Media score: {score_mean:.4f} | Std: {score_std:.4f}")
    print(f"📈 Máx score: {score_max:.4f}")
    print(f"🔧 Umbral final aplicado: {umbral_final:.4f}")
    print("📄 Resultados filtrados:")
    for i, (_, row) in enumerate(top_n.iterrows(), 1):
        print(f"{i}. {row['Nombre']:<20} | Score: {row['score']:.4f}")

    campos_salida = [
        "ID",
        "Nombre",
        "Descripcion",
        "Categoria",
        "Costo",
        "Fecha_Caducidad",
        "Stock",
    ]
    respuesta_cruda = top_n[campos_salida].to_dict(orient="records")

    # Generar prompt y enviar a Gemini
    prompt_llm = construir_prompt_llm(pregunta, respuesta_cruda)
    respuesta_llm = obtener_respuesta_llm(prompt_llm)

    return {
        "pregunta": pregunta,
        "resultados": respuesta_cruda,
        "respuesta_llm": respuesta_llm
    }
