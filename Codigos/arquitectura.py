from  ultralytics import YOLO

model=YOLO(r"runs/segment/mi_modelo_epp_small/weights/best.pt")
model.info(detailed=True)