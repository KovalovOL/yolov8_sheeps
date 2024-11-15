from ultralytics import YOLO
from PIL import Image


model = YOLO("yolov8n.pt")


def count_sheep(image_path, confidence_threshold=0.5, path_to_save = None):
    results = model(image_path)
    
    filtered_boxes = []
    sheep_count = 0
    
    for box in results[0].boxes:
        class_id = int(box.cls)  # Class obj
        confidence = float(box.conf)  # Confidence
        
        if class_id == 18 and confidence >= confidence_threshold:
            filtered_boxes.append(box)
            sheep_count += 1
    
    results[0].boxes = filtered_boxes
    
    annotated_image = results[0].plot()
    Image.fromarray(annotated_image).show()

    return sheep_count

image_path = "img\img1.jpg"
confidence_threshold = 0.6 

sheep_count = count_sheep(image_path, confidence_threshold)
print(f"Count of sheeps: {sheep_count}")
