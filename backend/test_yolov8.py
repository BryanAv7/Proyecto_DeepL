from ultralytics import YOLO

try:
    model = YOLO('C:/Users/USUARIO/Desktop/Proyecto_IA/modelos/bestV1.pt')
    print("✅ Modelo YOLOv8 cargado correctamente")
except Exception as e:
    print("❌ Error cargando YOLOv8:", e)
