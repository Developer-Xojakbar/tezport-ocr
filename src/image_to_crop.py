import io
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


# Инициализация YOLO модели для детекции номеров автомобилей (CAR_NUMBER)
# Используем специализированную модель для детекции номерных знаков
yolo_model = None

def _init_yolo_model():
    """Инициализирует YOLO модель для детекции номеров автомобилей."""
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    
    try:
        # Проверяем переменную окружения для пути к модели
        custom_model_path = os.environ.get('YOLO_LICENSE_PLATE_MODEL')
        
        # Пытаемся загрузить специализированную модель для детекции номерных знаков
        # Попробуем несколько вариантов моделей
        model_paths = []
        
        # Если указана кастомная модель, пробуем её первой
        if custom_model_path and Path(custom_model_path).exists():
            model_paths.append(custom_model_path)
        
        # Добавляем стандартные пути
        # Попробуем найти специализированные модели для номерных знаков
        # В первую очередь используем локальную модель пользователя
        base_dir = Path(__file__).resolve().parent.parent
        local_model_path = base_dir / "src/yolo_car_number.pt"
        
        if local_model_path.exists():
            model_paths.append(str(local_model_path))
            print(f"ℹ️ Найдена локальная модель: {local_model_path}")
        
        # Добавляем другие возможные пути
        model_paths.extend([
            "yolo_car_number.pt",  # В текущей директории
            "koushik-ai/yolov8-license-plate-detection/best.pt",  # Hugging Face модель
            'license_plate_detector.pt',  # Альтернативные имена
            'yolov8_license_plate.pt',
            'license_plate_yolov8.pt',
            'yolov8n.pt',  # Общая модель YOLOv8n как fallback
        ])
        
        # Если специализированная модель не найдена, пробуем загрузить локальную модель
        if local_model_path.exists():
            try:
                yolo_model = YOLO(str(local_model_path))
                print(f"✅ YOLO модель загружена: {local_model_path}")
                return yolo_model
            except Exception as e:
                print(f"⚠️ Ошибка загрузки локальной модели: {e}")
        
    except Exception as e:
        print(f"⚠️ Ошибка загрузки YOLO модели: {e}")
        print("ℹ️ Установите ultralytics: pip install ultralytics")
        return None

# Инициализируем модель при импорте
_init_yolo_model()


def image_to_crop(
    image_path: Union[str, Path, io.BytesIO],
    confidence: float = 0.25,
    save_to_output: bool = True,
) -> Optional[Union[str, io.BytesIO]]:
    """
    Обнаруживает номер автомобиля (CAR_NUMBER) на изображении с помощью YOLO и обрезает его.
    
    Args:
        image_path: Путь к изображению или BytesIO объект
        confidence: Минимальный порог уверенности для детекции (0.0-1.0)
        save_to_output: Сохранять ли обрезанное изображение в папку output
    
    Returns:
        Путь к обрезанному изображению (str) или BytesIO объект, или None, если номер не найден
    """
    # Убеждаемся, что модель загружена
    model = _init_yolo_model()
    if model is None:
        print("⚠️ YOLO модель не загружена. Установите ultralytics: pip install ultralytics")
        return None
    
    # Загрузка изображения
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
    else:
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"⚠️ Изображение не найдено: {image_path}")
            return None
        img = Image.open(img_path)
        img_array = np.array(img.convert('RGB'))
    
    # Детекция объектов с помощью YOLO
    results = model(img_array, conf=confidence, verbose=False)
    
    if not results or len(results) == 0:
        print("⚠️ Объекты не обнаружены на изображении")
        return None
    
    # Получаем первое изображение из результатов
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print("⚠️ Номер автомобиля не обнаружен")
        return None
    
    # Получаем классы детектированных объектов
    boxes = result.boxes.xyxy.cpu().numpy()  # Координаты в формате [x1, y1, x2, y2]
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
    
    if len(boxes) == 0:
        print("⚠️ Номер автомобиля не обнаружен")
        return None
    
    # Получаем имена классов модели (если доступны)
    class_names = model.names if hasattr(model, 'names') else None
    
    # Фильтруем боксы для поиска номерных знаков
    # ВАЖНО: Исключаем класс "car" и ищем только маленькие прямоугольные области
    filtered_boxes = []
    filtered_confidences = []
    img_width, img_height = img.size
    
    # Классы, которые нужно исключить (автомобили и другие большие объекты)
    excluded_classes = ['car', 'truck', 'bus', 'motorcycle', 'vehicle']
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        img_area = img_width * img_height
        
        # ИСКЛЮЧАЕМ большие объекты (автомобили) - они занимают слишком много места
        # Номерной знак должен быть маленькой частью изображения (обычно < 5%)
        if area > (img_area * 0.05):  # Исключаем области больше 5% изображения
            continue
        
        # Фильтруем слишком маленькие области (меньше 0.1% изображения)
        if area < (img_area * 0.001):
            continue
        
        # Номерные знаки обычно широкие прямоугольники (ширина значительно больше высоты)
        aspect_ratio = width / height if height > 0 else 0
        
        # Проверяем класс объекта (если доступен)
        is_license_plate = False
        is_excluded_class = False
        
        if classes is not None and class_names is not None:
            class_id = int(classes[i])
            class_name = class_names.get(class_id, '').lower()
            
            # ИСКЛЮЧАЕМ классы автомобилей
            if any(excluded in class_name for excluded in excluded_classes):
                is_excluded_class = True
                continue
            
            # Ищем классы связанные с номерными знаками
            if 'plate' in class_name or 'license' in class_name or 'number' in class_name:
                is_license_plate = True
        
        # Если это исключенный класс, пропускаем
        if is_excluded_class:
            continue
        
        # Номерные знаки обычно имеют соотношение сторон от 1.5:1 до 6:1 (широкие)
        # И должны быть относительно маленькими (не весь автомобиль)
        if is_license_plate or (1.5 <= aspect_ratio <= 6.0 and area < img_area * 0.05):
            filtered_boxes.append(box)
            filtered_confidences.append(confidences[i])
    
    if len(filtered_boxes) == 0:
        print("⚠️ YOLO не нашел номерной знак. Пробую использовать PaddleOCR как fallback...")
        # Fallback: используем PaddleOCR для поиска текстовых областей, похожих на номера
        try:
            from src.image_to_text import ocr_instance
            
            # Используем PaddleOCR для детекции текста
            ocr_results = ocr_instance.predict(input=img_array)
            
            if ocr_results:
                text_boxes = []
                for res in ocr_results:
                    if isinstance(res, dict):
                        bboxes = res.get("dt_polys", []) or res.get("boxes", [])
                        texts = res.get("rec_texts", [])
                        scores = res.get("rec_scores", [])
                    else:
                        bboxes = getattr(res, "dt_polys", None) or getattr(res, "boxes", None) or []
                        texts = getattr(res, "rec_texts", None) or []
                        scores = getattr(res, "rec_scores", None) or []
                    
                    if bboxes and texts:
                        for i, (bbox, text) in enumerate(zip(bboxes, texts)):
                            if not text or len(text) < 3:  # Номера обычно содержат минимум 3 символа
                                continue
                            
                            # Проверяем, похож ли текст на номер (буквы и цифры)
                            if any(c.isalnum() for c in text):
                                # Преобразуем полигон в прямоугольник
                                if bbox and len(bbox) >= 4:
                                    x_coords = [point[0] for point in bbox if len(point) >= 2]
                                    y_coords = [point[1] for point in bbox if len(point) >= 2]
                                    
                                    if x_coords and y_coords:
                                        x1, y1 = min(x_coords), min(y_coords)
                                        x2, y2 = max(x_coords), max(y_coords)
                                        
                                        width = x2 - x1
                                        height = y2 - y1
                                        area = width * height
                                        img_area = img_width * img_height
                                        aspect_ratio = width / height if height > 0 else 0
                                        
                                        # Фильтруем по размеру и форме номерного знака
                                        if (0.001 <= area / img_area <= 0.05 and 
                                            1.5 <= aspect_ratio <= 6.0):
                                            text_boxes.append({
                                                'box': [x1, y1, x2, y2],
                                                'text': text,
                                                'score': scores[i] if scores and i < len(scores) else 0.5
                                            })
                
                if text_boxes:
                    # Выбираем область с наибольшей уверенностью или наибольшей площадью
                    best_box = max(text_boxes, key=lambda x: x['score'])
                    selected_box = best_box['box']
                    print(f"✅ Номерной знак найден через PaddleOCR: {best_box['text']}")
                else:
                    print("⚠️ Номерной знак не обнаружен ни YOLO, ни PaddleOCR.")
                    print("   Рекомендуется использовать специализированную YOLO модель для номерных знаков.")
                    return None
            else:
                print("⚠️ Номерной знак не обнаружен.")
                return None
        except Exception as e:
            print(f"⚠️ Ошибка при использовании PaddleOCR fallback: {e}")
            print("⚠️ Номерной знак не обнаружен.")
            return None
    
    # Выбираем bounding box с наибольшей уверенностью (для номерных знаков это важнее площади)
    max_conf_idx = np.argmax(filtered_confidences)
    selected_box = filtered_boxes[max_conf_idx]
    
    print(f"✅ Номер автомобиля обнаружен с уверенностью: {filtered_confidences[max_conf_idx]:.2f}")
    x1, y1, x2, y2 = map(int, selected_box)
    
    # Добавляем небольшой отступ для лучшего обрезания
    padding = 10
    img_width, img_height = img.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_width, x2 + padding)
    y2 = min(img_height, y2 + padding)
    
    # Обрезаем изображение
    cropped_img = img.crop((x1, y1, x2, y2))
    
    if save_to_output:
        # Создаем папку output если её нет
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Генерируем имя файла
        if isinstance(image_path, io.BytesIO):
            base_name = "cropped_car_number"
        else:
            img_path = Path(image_path)
            base_name = img_path.stem
        
        output_path = output_dir / f"{base_name}_car_number.jpg"
        
        # Сохраняем обрезанное изображение
        cropped_img.save(output_path, "JPEG", quality=95)
        print(f"✅ Обрезанное изображение сохранено: {output_path}")
        return str(output_path)
    else:
        # Возвращаем BytesIO объект
        buffer = io.BytesIO()
        cropped_img.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        return buffer
