# Librerias por utilizar
from fastapi import FastAPI, File, UploadFile, HTTPException
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

from ultralytics import YOLO  

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir la arquitectura SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Variables globales para los modelos
yolo_model = None
cnn_model = None
cnn_transform = None
id2label = None
productos_df = None  # ✅ DataFrame con los datos de productos

@app.on_event("startup")
async def load_models():
    global yolo_model, cnn_model, cnn_transform, id2label, productos_df

    try:
        # Cargar modelo YOLOv8
        yolo_model = YOLO('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/last.pt')  
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
        cnn_model = SimpleCNN(num_classes=num_classes)
        checkpoint = torch.load('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/modelo_productos.pth', map_location=torch.device('cpu'))
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
        productos_df = pd.read_csv('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/data_actualizadoFinal.csv')
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
            etiqueta_cnn = await predict_product_id(crop_rgb)

            # Buscar en el DataFrame
            producto_info = productos_df[productos_df['ID'] == etiqueta_cnn]
            if not producto_info.empty:
                producto_dict = producto_info.iloc[0].to_dict()
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
