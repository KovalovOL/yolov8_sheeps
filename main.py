from ultralytics import YOLO
from PIL import Image


model = YOLO("yolov8n.pt")


def count_sheep(image_path, confidence_threshold=0.5, show_res = False):
    """
    return count of animals in img

    show_res - Need to show image result
    """

    results = model(image_path)
    
    filtered_boxes = []
    sheep_count = 0
    
    for box in results[0].boxes:
        class_id = int(box.cls)  # Class obj
        confidence = float(box.conf)  # Confidence
        
        if class_id == 18 and confidence >= confidence_threshold:
            filtered_boxes.append(box)
            sheep_count += 1
    
    if show_res:
        results[0].boxes = filtered_boxes
        annotated_image = results[0].plot()
        Image.fromarray(annotated_image).show()

    return sheep_count



def get_sheep_coordinates(image_path, confidence_threshold=0.5):
    """
    return list of coordinates
    """

    results = model(image_path)
    
    sheep_coords = []
    
    for box in results[0].boxes:
        class_id = int(box.cls)  # Класс объекта
        confidence = float(box.conf)  # Уверенность модели
        if class_id == 18 and confidence >= confidence_threshold:     
            coords = box.xyxy.numpy().tolist()[0]
            coords = [round(c, 2) for c in coords]

            sheep_coords.append(coords)
    return sheep_coords



image_path = "img/img1.jpg"
confidence_threshold = 0.6 

sheep_count = count_sheep(image_path, confidence_threshold, True)
print(f"Count of sheeps: {sheep_count}")
print(get_sheep_coordinates(image_path, confidence_threshold))
