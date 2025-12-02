from ultralytics import YOLO
data = r"C:\Users\tomas\IA-CNN-Bounding-Boxes-EPP\Dataset PPE -Segmentation-.v1-first.yolov8\data.yaml" #direccion del dataset correspondiente 
def main():
   
    model = YOLO('yolov8s-seg.pt')#aca dependiendo el modelo que quiere lo cambio 


    print("Iniciando entrenamiento...")
    results = model.train(
        data=data ,
        epochs=50,                
        imgsz=640,                
        batch=8,                  
        name='mi_modelo_epp_small',
        device='0'                
    )

if __name__ == '__main__':
    main()