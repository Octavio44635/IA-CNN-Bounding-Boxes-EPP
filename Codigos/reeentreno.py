from sympy import true
from ultralytics import YOLO
data = r"C:\Users\tomas\IA-CNN-Bounding-Boxes-EPP\Dataset PPE -Segmentation-.v1-first.yolov8\data.yaml"
def continuar_entrenamiento():
    # En lugar de 'yolov8n-seg.pt', cargamos 'last.pt'
    # Ajusta la ruta 
    model = YOLO(r"runs/segment/mi_modelo_epp_simi_ad_hoc/weights/last.pt")

    # El comando clave aquí es resume=True
    model.train(
               
        resume= "true"
    )

if __name__ == '__main__':
    continuar_entrenamiento()