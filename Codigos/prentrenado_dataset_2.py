from ultralytics import YOLO
data = r"C:\Users\tomas\Desktop\PPE Final.v6-no-preprocessing-augmentations-70-10-20.yolov8\data.yaml"
def main():
    # 1. Cargar el modelo base
    # Usamos la versión "Nano" de segmentación para empezar (es rápido)
    model = YOLO('yolov8n-seg.pt')

    # 2. Entrenar el modelo
    print("Iniciando entrenamiento...")
    results = model.train(
        data=data , # <--- REVISA QUE ESTA RUTA SEA CORRECTA
        epochs=50,                # 50 vueltas completas al dataset
        imgsz=640,                # Tamaño de imagen estándar
        batch=8,                  # Cuántas fotos procesa a la vez (bájalo si te quedas sin memoria)
        name='mi_modelo_epp_nano_dataset_2',     # Nombre de la carpeta de resultados
        device='0'                # Usa '0' si tienes GPU NVIDIA, o 'cpu' si no tienes
    )

if __name__ == '__main__':
    main()