# 🧠 Reconocimiento Automático de Productos

Esta aplicación permite identificar productos a partir de imágenes usando una combinación de modelos. 
Utiliza **YOLOv8** para detectar el producto en la imagen, y un Modelo **CNN personalizado** para clasificar el tipo exacto de producto. 
El sistema está compuesto por un **backend en FastAPI** y un **frontend en Angular**.

---

## 📁 Estructura del Proyecto

```
Proyecto_IA/
│
├── backend/               # Servidor FastAPI con la lógica de detección y predicción
│   ├── main.py            # Código principal del backend
│   └── productos_codificados.csv  # Base de datos de productos
│
├── frontend/              # Aplicación web construida con Angular
│   └── src/app/dashboard  # Componente de carga y predicción
│
├── modelos/               # Modelos entrenados (YOLOv8 y CNN)
│   ├── bestV1.pt          # Modelo de detección YOLOv8
│   └── modelo_productos.pth  # Modelo de clasificación CNN
│
├── testeoImagenes/        # Carpeta para pruebas de imágenes (opcional)
└── README.md              # Instrucciones y documentación del proyecto
```

---

## ⚙️ Requisitos

### 🔧 Backend (FastAPI)

- Python 3.10 o superior
- FastAPI
- Uvicorn
- PyTorch
- OpenCV
- Ultralytics (YOLOv8)
- Pandas
- Pillow

Instala las dependencias ejecutando:

```bash
cd backend
pip install -r requirements.txt
```

> Si no tienes un archivo `requirements.txt`, puedes instalar manualmente:

```bash
pip install fastapi uvicorn torch torchvision opencv-python pandas pillow ultralytics
```
> [!IMPORTANT]
> Es importante saber que para la *nueva* integracion del Asitente se necesita nuevas importaciones y una API-KEY que se tiene que configurar de manera manual por seguridad.
>
> Se esta usando el LLM de GEMINI
>
>Modelo: gemini-1.5-flash
> 
> Librerias:
> ```
> pip install sentence-transformers
> pip install google-generativeai
> ```

---

### 🌐 Frontend (Angular)

- Node.js (v16+)
- Angular CLI

Instala las dependencias con:

```bash
cd frontend
npm install
```


---

### 💻 Modelo de Detección y OCR de Productos

Este proyecto implementa un modelo de detección de objetos basado en YOLOv8, entrenado con un dataset fusionado entre:

• El dataset público SKU110K.

• Un dataset personalizado, etiquetado manualmente en Roboflow, con imágenes de escaparates reales.


El objetivo es detectar productos en estanterías de tiendas y extraer información textual relevante como nombre, marca, sabor, precio o fecha de caducidad mediante OCR.


---

🗃 Estructura del dataset fusionado

dataset_fusionado/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/


---

🔍 Implementación de OCR

La extracción de texto se realiza sobre los productos recortados (bounding boxes) utilizando la API de Google Cloud Vision OCR, lo que permite obtener resultados más precisos que herramientas offline tradicionales, incluso en condiciones de imagen desafiantes.

El OCR se aplica automáticamente sobre todas las imágenes recortadas, y los textos detectados se almacenan en archivos .csv y .json para facilitar su análisis posterior.


---


## 🚀 ¿Cómo ejecutar la aplicación?

### 1. Iniciar el servidor del backend (FastAPI)

Desde la carpeta `backend`:

```bash
uvicorn main:app --reload
```

Esto levantará el servidor FastAPI en:

```
http://127.0.0.1:8000
```

Puedes probar el backend con:

- `http://127.0.0.1:8000/docs` → Documentación automática (Swagger)
- `http://127.0.0.1:8000/api/saludo` → Endpoint de prueba

---

### 2. Iniciar el servidor del frontend (Angular)

Desde la carpeta `frontend`:

```bash
ng serve
```

Esto abrirá automáticamente la aplicación en:

```
http://localhost:4200
```

---

## 🧪 ¿Qué hace la aplicación?

1. El usuario carga una imagen de un producto desde la web.
2. La imagen se envía al backend para ser procesada.
3. **YOLOv8** detecta el producto y lo recorta.
4. **CNN** clasifica el producto entre varias clases entrenadas.
5. El frontend muestra toda la información: nombre, descripción, costo, categoría, stock y fecha de caducidad.

---

## 📦 Visualización de los Resultados

```
ID: 93D361A0
Nombre: Pepsi Lata
Descripción: Bebida gaseosa sabor cola, lata de 355ml
Categoría: Bebidas
Costo: $0.85
Stock: 95 unidades
Fecha de Caducidad: 2025-11-25
```

