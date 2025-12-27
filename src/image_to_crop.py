import io
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


yolo_model = None
yolo_container_model = None
_base_dir = Path(__file__).resolve().parent.parent

def _init_yolo_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    
    try:
        custom_model_path = os.environ.get('YOLO_LICENSE_PLATE_MODEL')
        model_paths = []
        
        if custom_model_path and Path(custom_model_path).exists():
            model_paths.append(custom_model_path)
        
        model_paths.extend([
            str(_base_dir / "src/yolo_car_number.pt"),
        ])
        
        for model_path in model_paths:
            try:
                yolo_model = YOLO(model_path)
                return yolo_model
            except Exception:
                continue
        
        return yolo_model
    except Exception:
        return None

_init_yolo_model()


def image_to_car_number_crop(
    image_path: Union[str, Path, io.BytesIO],
    confidence: float = 0.25,
    save_to_output: bool = False,
) -> Optional[Union[str, io.BytesIO]]:
    model = _init_yolo_model()
    if model is None:
        return None
    
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
    else:
        img_path = Path(image_path)
        if not img_path.exists():
            return None
        img = Image.open(img_path)
        img_array = np.array(img.convert('RGB'))
    
    results = model(img_array, conf=confidence, verbose=False)
    
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return None
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
    
    if len(boxes) == 0:
        return None
    
    class_names = model.names if hasattr(model, 'names') else None
    filtered_boxes = []
    filtered_confidences = []
    img_width, img_height = img.size
    img_area = img_width * img_height
    excluded_classes = {'car', 'truck', 'bus', 'motorcycle', 'vehicle'}
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area > (img_area * 0.05) or area < (img_area * 0.001):
            continue
        
        aspect_ratio = width / height if height > 0 else 0
        
        if classes is not None and class_names is not None:
            class_id = int(classes[i])
            class_name = class_names.get(class_id, '').lower()
            
            if any(excluded in class_name for excluded in excluded_classes):
                continue
            
            if 'plate' in class_name or 'license' in class_name or 'number' in class_name:
                filtered_boxes.append(box)
                filtered_confidences.append(confidences[i])
                continue
        
        if 1.5 <= aspect_ratio <= 6.0:
            filtered_boxes.append(box)
            filtered_confidences.append(confidences[i])
    
    if len(filtered_boxes) == 0:
        return None
    
    max_conf_idx = np.argmax(filtered_confidences)
    selected_box = filtered_boxes[max_conf_idx]
    x1, y1, x2, y2 = map(int, selected_box)
    
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_width, x2 + padding)
    y2 = min(img_height, y2 + padding)
    
    cropped_img = img.crop((x1, y1, x2, y2))
    
    if cropped_img.mode != 'RGB':
        cropped_img = cropped_img.convert('RGB')
    
    if save_to_output:
        output_dir = _base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        if isinstance(image_path, io.BytesIO):
            base_name = "cropped_car_number"
        else:
            base_name = Path(image_path).stem
        
        output_path = output_dir / f"{base_name}_car_number.jpg"
        cropped_img.save(output_path, "JPEG", quality=95)
        return str(output_path)
    else:
        buffer = io.BytesIO()
        cropped_img.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        return buffer

def _init_container_yolo_model():
    global yolo_container_model
    if yolo_container_model is not None:
        return yolo_container_model
    
    try:
        custom_model_path = os.environ.get('YOLO_CONTAINER_MODEL')
        model_paths = []
        
        if custom_model_path and Path(custom_model_path).exists():
            model_paths.append(custom_model_path)
        
        model_paths.extend([
            str(_base_dir / "src/yolo_container_number.pt"),
        ])
        
        for model_path in model_paths:
            try:
                yolo_container_model = YOLO(model_path)
                return yolo_container_model
            except Exception:
                continue
        
        return yolo_container_model
    except Exception:
        return None


def image_to_container_number_crop(
    image_path: Union[str, Path, io.BytesIO],
    confidence: float = 0.25,
    save_to_output: bool = False,
) -> Optional[Union[str, io.BytesIO]]:
    model = _init_container_yolo_model()
    if model is None:
        return None
    
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
    else:
        img_path = Path(image_path)
        if not img_path.exists():
            return None
        img = Image.open(img_path)
        img_array = np.array(img.convert('RGB'))
    
    results = model(img_array, conf=confidence, verbose=False)
    
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return None
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    if len(boxes) == 0:
        return None
    
    img_width, img_height = img.size
    img_area = img_width * img_height
    filtered_boxes = []
    filtered_confidences = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area > (img_area * 0.5) or area < (img_area * 0.001):
            continue
        
        aspect_ratio = width / height if height > 0 else 0
        
        if 1.0 <= aspect_ratio <= 10.0:
            filtered_boxes.append(box)
            filtered_confidences.append(confidences[i])
    
    if len(filtered_boxes) == 0:
        return None
    
    max_conf_idx = np.argmax(filtered_confidences)
    selected_box = filtered_boxes[max_conf_idx]
    x1, y1, x2, y2 = map(int, selected_box)
    
    box_width = x2 - x1
    box_height = y2 - y1
    is_horizontal = box_width > box_height
    
    if is_horizontal:
        padding_right = int(box_width * 1.0)
        padding_top = int(box_height * 0.75)
        padding_bottom = int(box_height * 1.5)
        padding_left = 10
        
        x1 = max(0, x1 - padding_left)
        y1 = max(0, y1 - padding_top)
        x2 = min(img_width, x2 + padding_right)
        y2 = min(img_height, y2 + padding_bottom)
    else:
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
    
    cropped_img = img.crop((x1, y1, x2, y2))
    
    if cropped_img.mode != 'RGB':
        cropped_img = cropped_img.convert('RGB')
    
    if save_to_output:
        output_dir = _base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        if isinstance(image_path, io.BytesIO):
            base_name = "cropped_container_number"
        else:
            base_name = img_path.stem if img_path else Path(image_path).stem
        
        output_path = output_dir / f"{base_name}_container_number.jpg"
        cropped_img.save(output_path, "JPEG", quality=95)
        return str(output_path)
    else:
        buffer = io.BytesIO()
        cropped_img.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        return buffer

def image_to_crop(
    image_path: Union[str, Path, io.BytesIO],
    confidence: float = 0.25,
    save_to_output: bool = False,
) -> Optional[Union[str, io.BytesIO]]:

    car_number_image = image_to_car_number_crop(image_path, confidence, save_to_output)
    if car_number_image:
        return {'detect': 'car', 'image': car_number_image}

    container_number_image = image_to_container_number_crop(image_path, confidence, save_to_output)
    if container_number_image:
        return {'detect': 'container', 'image': container_number_image}

    buffer = io.BytesIO()
    Image.new('RGB', (1, 1), color='white').save(buffer, "JPEG", quality=95)
    buffer.seek(0)
    return {'detect': None, 'image': buffer}