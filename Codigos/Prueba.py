from ultralytics import YOLO
import cv2
# # 1. CARGAR TU MODELO
# # Si tu entrenamiento sigue corriendo, puedes usar 'last.pt' en lugar de 'best.pt'
# # Ajusta la ruta si es necesario.
model_path = r"runs/segment/mi_modelo_epp_medium/weights/best.pt" 

try:
    model = YOLO(model_path)
    print(f"Modelo cargado exitosamente: {model_path}")
except:
    print(f"Error: No se encontró el modelo en {model_path}")
    print("Verifica que la ruta sea correcta.")
    exit()

# # 2. ELEGIR QUÉ PROBAR
# # Cambia 'source' por:
# #   - '0'             -> Para usar tu WEBCAM en vivo.
# #   - 'nombre_foto.jpg' -> Para una imagen especifica.
# #   - 'carpeta/fotos' -> Para probar todas las fotos de una carpeta.
# source = "0"#r"runs/prueba.jpg"  

# # 3. EJECUTAR LA PREDICCIÓN
# print("Ejecutando prueba... Presiona 'q' en la ventana para salir.")

# results = model.predict(
#     source=source,
#     show=True,      # Muestra la ventana con el video/foto
#     conf=0.5,       # Confianza mínima: solo muestra si está 50% seguro
#     save=True,      # Guarda el resultado en la carpeta 'runs/segment/predict'
#     imgsz=640       # Tamaño de la imagen
# )

# print("Prueba finalizada.")
if __name__ == '__main__':
    metrics = model.val()

    # Puedes acceder a valores específicos así:
    print(metrics.box.map)    # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.r)  
    print(metrics.box.p)  
    print(metrics.box.f1)  

