# from ultralytics import YOLO
# import cv2


# def training(model):
#     path = r"C:\Users\tomas\IA-CNN-Bounding-Boxes-EPP\Dataset PPE -Segmentation-.v1-first.yolov8\data.yaml"
#     results = model.train(data=path, epochs=100, imgsz=640, batch=8,name='mi_modelo_epp',device = 0, save=True, resume=False)


# def prediction(model):
#     # path = r"I"C:\Users\tomas\IA-CNN-Bounding-Boxes-EPP\Dataset PPE -Segmentation-.v1-first.yolov8""
#     resultados = model.predict(source=path, show=True, save=True)

#     # 3. Procesar los resultados
#     for r in resultados:
#         # Aquí tienes las Bounding Boxes
#         cajas = r.boxes.xyxy.cpu().numpy()  # coordenadas x1, y1, x2, y2
        
#         # Aquí tienes las Máscaras de Segmentación
#         mascaras = r.masks.data.cpu().numpy() # mapas de bits de las máscaras
        
#         print(f"Se detectaron {len(cajas)} objetos con sus respectivas máscaras.")




# # 1. Cargar el modelo preentrenado de segmentación
# # Automáticamente descargará el archivo si no lo tienes


# training(YOLO('yolov8x-seg.pt'))
# # prediction(YOLO('yolov8x-seg.pt'))

# # 2. Realizar la inferencia en una imagen
# # source puede ser '0' para webcam, una ruta de imagen, o un video



from ultralytics import YOLO
data = r"C:\Users\tomas\IA-CNN-Bounding-Boxes-EPP\Dataset PPE -Segmentation-.v1-first.yolov8\data.yaml"
def main():
    # 1. Cargar el modelo base
    # Usamos la versión "Nano" de segmentación para empezar (es rápido)
    model = YOLO('yolov8m-seg.pt')

    # 2. Entrenar el modelo
    print("Iniciando entrenamiento...")
    results = model.train(
        data=data , # <--- REVISA QUE ESTA RUTA SEA CORRECTA
        epochs=50,                # 50 vueltas completas al dataset
        imgsz=640,                # Tamaño de imagen estándar
        batch=8,                  # Cuántas fotos procesa a la vez (bájalo si te quedas sin memoria)
        name='mi_modelo_epp_v2',     # Nombre de la carpeta de resultados
        device='0'                # Usa '0' si tienes GPU NVIDIA, o 'cpu' si no tienes
    )

if __name__ == '__main__':
    main()