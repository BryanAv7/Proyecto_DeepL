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

from ultralytics import YOLO  
 

app = FastAPI()

# Configuracion RUTAS
RUTA_DATASET = "../modelos/data_actualizadoArreglado.csv"
RUTAS_MODELO_YOLO = "../modelos/last.pt"
RUTAS_MODELO_CNN = "../modelos/modelo_productosbeta.pth"
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

        for box in detections:
            class_idx = int(box.cls.item())
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

        # Obtener todas las detecciones YOLO
        results = yolo_model(img_cv, conf=0.20)
        detections = results[0].boxes

        if len(detections) == 0:
            raise HTTPException(status_code=400, detail="No se detectaron productos en la imagen")

        productos_detectados = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = img_cv[y1:y2, x1:x2]

            if crop.size == 0:
                print("⚠️ Detección ignorada por recorte vacío.")
                continue

            # Convertir recorte a RGB y clasificar con CNN
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Codificar imagen recortada como base64
            _, buffer = cv2.imencode('.jpg', crop_rgb)
            imagen_crop_base64 = base64.b64encode(buffer).decode('utf-8')

            etiqueta_cnn = await predict_product_id(crop_rgb)

            # Buscar en el DataFrame
            producto_info = productos_df[productos_df['ID'] == etiqueta_cnn]
            if not producto_info.empty:
                producto_dict = producto_info.iloc[0].to_dict()
                producto_dict["imagen_crop_base64"] = imagen_crop_base64  # ← Agregamos la imagen recortada
                productos_detectados.append(producto_dict)
            else:
                print(f"⚠️ Producto con ID {etiqueta_cnn} no encontrado en CSV.")

        if not productos_detectados:
            raise HTTPException(status_code=404, detail="No se encontró información de los productos detectados.")

        # ✅ Devolver la lista de productos clasificados
        return {"productos": productos_detectados}

    except Exception as e:
        print(f"❌ Error general en predict-product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
