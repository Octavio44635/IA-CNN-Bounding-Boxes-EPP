from sympy import true
from ultralytics import YOLO
import cv2
# # 1. CARGAR TU MODELO
# # Si tu entrenamiento sigue corriendo, puedes usar 'last.pt' en lugar de 'best.pt'
# # Ajusta la ruta si es necesario.
model_path = r"runs/segment/mi_modelo_epp_simi_ad_hoc/weights/best.pt" 

try:
    model = YOLO(model_path)
    print(f"Modelo cargado exitosamente: {model_path}")
except:
    print(f"Error: No se encontró el modelo en {model_path}")
    print("Verifica que la ruta sea correcta.")
    exit()

# 2. ELEGIR QUÉ PROBAR
# Cambia 'source' por:
#   - '0'             -> Para usar tu WEBCAM en vivo.
#   - 'nombre_foto.jpg' -> Para una imagen especifica.
#   - 'carpeta/fotos' -> Para probar todas las fotos de una carpeta.
source =r"fotos octi/tomi_civil.png"  

# 3. EJECUTAR LA PREDICCIÓN
print("Ejecutando prueba... Presiona 'q' en la ventana para salir.")

results = model.predict(
    source=source,
    show=False,
    conf=0.5,
    save=True,
    imgsz=640,
    device='0',
    
    # --- AGREGA ESTO ---
    project='Ultimos pruebas',      # Carpeta principal
    name='predict',    # Subcarpeta
    exist_ok=True            # ¡CLAVE! Guarda todo ahí sin crear copias
    
)
