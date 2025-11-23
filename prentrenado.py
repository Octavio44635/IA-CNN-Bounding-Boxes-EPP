from ultralytics import YOLO
import cv2

# 1. Cargar el modelo preentrenado de segmentación
# Automáticamente descargará el archivo si no lo tienes
model = YOLO('yolov8n-seg.pt')

# 2. Realizar la inferencia en una imagen
# source puede ser '0' para webcam, una ruta de imagen, o un video
path = r"A:\Escritorio\Facultad\Cuatri\TP_IA\IA-CNN-Bounding-Boxes-EPP\runs\prueba.jpg"
resultados = model.predict(source=path, show=True, save=True)

# 3. Procesar los resultados
for r in resultados:
    # Aquí tienes las Bounding Boxes
    cajas = r.boxes.xyxy.cpu().numpy()  # coordenadas x1, y1, x2, y2
    
    # Aquí tienes las Máscaras de Segmentación
    mascaras = r.masks.data.cpu().numpy() # mapas de bits de las máscaras
    
    print(f"Se detectaron {len(cajas)} objetos con sus respectivas máscaras.")