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
        yolo_model = YOLO('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/bestV1.pt')  
        print("✅ Modelo YOLOv8 cargado correctamente")
        
        # Definir el mapeo de etiquetas (ajusta según tu modelo)
        id2label = {
            '85F3F915': 0, '82B09699': 1, '0A99042A': 2, '5A5F332D': 3,
            '09D66912': 4, '25F069F7': 5, 'AA878A53': 6, 'C4739F50': 7,
            '995BA078': 8, '5DC28BDC': 9, '93D361A0': 10, 'FFFF9368': 11,
            'B2B29A8A': 12, 'B074ABAC': 13, '29A5080B': 14
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
        productos_df = pd.read_csv('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/productos_codificados.csv')
        print("✅ Datos de productos cargados correctamente")

    except Exception as e:
        print(f"❌ Error al cargar los modelos o CSV: {str(e)}")
        raise RuntimeError(f"Error al cargar los modelos: {str(e)}")

async def get_most_confident_detection(image_bytes):
    try:
        # Convertir bytes a imagen OpenCV
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Inferencia con YOLOv8
        results = yolo_model(img_cv, conf=0.01)
        detections = results[0].boxes  

        if len(detections) == 0:
            print("⚠️ YOLO no detectó ningún objeto con la confianza mínima.")
            return None, None

        # Obtener la detección con mayor confianza
        best_box = max(detections, key=lambda b: b.conf)
        print(f"🎯 Mejor detección con confianza: {best_box.conf.item():.4f}")

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # Recortar la región detectada
        crop = img_cv[y1:y2, x1:x2]
        if crop.size == 0:
            print("⚠️ La región recortada está vacía.")
            return None, None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), None

    except Exception as e:
        print(f"❌ Error en procesamiento YOLO: {str(e)}")
        raise RuntimeError(f"Error en procesamiento YOLO: {str(e)}")

async def predict_product_id(image_crop):
    try:
        # Convertir a PIL Image
        img_pil = Image.fromarray(image_crop)

        # Aplicar transformaciones
        input_tensor = cnn_transform(img_pil).unsqueeze(0)

        # Predicción con CNN
        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_id = pred.item()

        # Convertir ID numérico a etiqueta (string)
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
        # Leer imagen subida
        image_bytes = await file.read()

        # Obtener el recorte más confiable con YOLO
        image_crop, _ = await get_most_confident_detection(image_bytes)
        if image_crop is None:
            raise HTTPException(status_code=400, detail="No se detectaron productos en la imagen")

        # Predecir el ID del producto con CNN
        product_id = await predict_product_id(image_crop)

        # ✅ Buscar en el DataFrame usando la columna 'ID'
        producto_info = productos_df[productos_df['ID'] == product_id]
        if producto_info.empty:
            raise HTTPException(status_code=404, detail="Producto no encontrado en la base de datos")

        # ✅ Convertir a diccionario y retornar
        producto_dict = producto_info.iloc[0].to_dict()
        return producto_dict

    except Exception as e:
        print(f"❌ Error general en predict-product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/saludo")
def obtener_saludo():
    return {"mensaje": "🚀¡Aplicación de Inteligencia Artificial para el Reconocimiento de Productos!"}
