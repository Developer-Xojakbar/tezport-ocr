import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR

# def _check_gpu_available() -> bool:
#     try:
#         import paddle
#         if not paddle.device.is_compiled_with_cuda():
#             return False
#         try:
#             gpu_count = paddle.device.cuda.device_count()
#             if gpu_count > 0:
#                 paddle.device.set_device('gpu:0')
#                 return True
#             return False
#         except Exception:
#             return False
#     except ImportError:
#         return False
#     except Exception:
#         return False

# USE_GPU = _check_gpu_available()

# if USE_GPU:
#     print("✅ GPU обнаружен! PaddleOCR будет использовать GPU для ускорения.")
# else:
#     print("ℹ️ GPU не обнаружен или недоступен. Используется CPU.")

ocr_instance = PaddleOCR(
    lang="en",
    # use_gpu=USE_GPU,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_angle_cls=True,
)


def _group_texts_by_line(
    texts: List[str],
    scores: List[float],
    bboxes: List[List[List[int]]],
    line_threshold: float = 0.5,
) -> Tuple[List[str], List[float]]:
    if not texts or not bboxes or len(texts) != len(bboxes):
        return texts, scores
    
    text_items = []
    for i, (text, score, bbox) in enumerate(zip(texts, scores, bboxes)):
        if not text:
            continue
        
        if bbox is None:
            continue
        
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        
        if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
            continue
        
        try:
            first_point = bbox[0]
            if not isinstance(first_point, (list, tuple, np.ndarray)) or len(first_point) < 2:
                continue
            
            points = []
            for point in bbox:
                if isinstance(point, np.ndarray):
                    point = point.tolist()
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append(point)
            
            if not points:
                continue
            
            y_coords = [point[1] for point in points]
            x_coords = [point[0] for point in points]
            
            avg_y = sum(y_coords) / len(y_coords)
            height = max(y_coords) - min(y_coords)
            min_x = min(x_coords)
            
            text_items.append({
                'text': text,
                'score': score,
                'y': avg_y,
                'height': height,
                'x': min_x,
            })
        except (IndexError, TypeError, ValueError, AttributeError):
            continue
    
    if not text_items:
        return texts, scores
    
    text_items.sort(key=lambda x: x['y'])
    
    grouped_texts = []
    grouped_scores = []
    current_line = []
    current_line_y = None
    current_line_height = 0
    
    for item in text_items:
        if current_line_y is None:
            current_line = [item]
            current_line_y = item['y']
            current_line_height = item['height']
        else:
            y_diff = abs(item['y'] - current_line_y)
            threshold = max(current_line_height, item['height']) * line_threshold
            
            if y_diff <= threshold:
                current_line.append(item)
                current_line_height = max(current_line_height, item['height'])
            else:
                if current_line:
                    current_line.sort(key=lambda x: x['x'])
                    combined_text = ' '.join([item['text'] for item in current_line])
                    avg_score = sum([item['score'] for item in current_line]) / len(current_line)
                    grouped_texts.append(combined_text)
                    grouped_scores.append(avg_score)
                
                current_line = [item]
                current_line_y = item['y']
                current_line_height = item['height']
    
    if current_line:
        current_line.sort(key=lambda x: x['x'])
        combined_text = ' '.join([item['text'] for item in current_line])
        avg_score = sum([item['score'] for item in current_line]) / len(current_line)
        grouped_texts.append(combined_text)
        grouped_scores.append(avg_score)
    
    return grouped_texts, grouped_scores


def _enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    """Улучшает изображение для лучшего распознавания текста (черный/белый)."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_gray = img.convert('L')
    
    img_array = np.array(img_gray)
    mean_brightness = np.mean(img_array)
    
    if mean_brightness < 128:
        img_array = 255 - img_array
    
    img_enhanced = Image.fromarray(img_array.astype(np.uint8))
    
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(2.0)
    
    img_enhanced = ImageOps.autocontrast(img_enhanced, cutoff=5)
    
    img_rgb = img_enhanced.convert('RGB')
    return img_rgb


def image_to_text(
    image_path: Union[str, Path, io.BytesIO],
    min_score: float = 0.6,
    group_by_line: bool = True,
    line_threshold: float = 0.5,
) -> Dict[str, List]:
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)
    
    img = _enhance_image_for_ocr(img)
    
    img_array = np.array(img)
    results = ocr_instance.predict(input=img_array)

    rec_texts: List[str] = []
    rec_scores: List[float] = []
    rec_bboxes: List[List[List[int]]] = []

    if results:
        for res in results:
            if isinstance(res, dict):
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                bboxes = res.get("dt_polys", []) or res.get("boxes", [])
            else:
                texts = getattr(res, "rec_texts", None) or []
                scores = getattr(res, "rec_scores", None) or []
                bboxes = getattr(res, "dt_polys", None) or getattr(res, "boxes", None) or []

            if texts:
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    
                    score = 0.0
                    if scores and i < len(scores):
                        score = scores[i]
                    
                    bbox = []
                    if bboxes and i < len(bboxes):
                        bbox = bboxes[i]
                    
                    if score >= min_score:
                        rec_texts.append(text)
                        rec_scores.append(score)
                        rec_bboxes.append(bbox)

    if group_by_line and rec_texts and rec_bboxes:
        rec_texts, rec_scores = _group_texts_by_line(
            rec_texts, rec_scores, rec_bboxes, line_threshold
        )

    return {
        "data": {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        },
        "texts": rec_texts,
    }

