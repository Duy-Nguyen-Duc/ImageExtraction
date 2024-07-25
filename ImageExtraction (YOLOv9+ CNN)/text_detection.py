from ultralytics import YOLO

model = YOLO('yolov8s.yaml').load('yolov8s.pt')

epochs = 200
imgsz = 1024

results = model.train(
    data = './yolo_data/data.yaml',
    epochs = epochs,
    project = 'models',
    name = 'yolov8/detect/train'
)