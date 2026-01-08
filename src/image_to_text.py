import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from paddleocr import PaddleOCR

def _check_gpu_available() -> bool:
    try:
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            return False
        try:
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                paddle.device.set_device('gpu:0')
                return True
            return False
        except Exception:
            return False
    except ImportError:
        return False
    except Exception:
        return False

USE_GPU = _check_gpu_available()

if USE_GPU:
    print("✅ GPU обнаружен! PaddleOCR будет использовать GPU для ускорения.")
else:
    print("ℹ️ GPU не обнаружен или недоступен. Используется CPU.")

ocr_instance = PaddleOCR(
    lang="en",
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

# 50 -> 3
# 60 -> 3 xam yaxshi
# 75 -> 2: lekin yaxshiroq
image_quality = 75
def _enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Улучшает изображение для лучшего распознавания текста
    (особенно для чёрных цифр внутри цветных/светлых квадратов и на
    неоднородном фоне), без жёсткой привязки к белому цвету.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Нормализуем ориентацию (на случай EXIF-поворота)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Перевод в градации серого
    img_gray = img.convert('L')

    # Увеличиваем картинку, чтобы тонкие штрихи цифр были толще
    w, h = img_gray.size
    min_side = min(w, h)
    if min_side < 800:
        # масштаб подбираем мягко, чтобы не размывать слишком сильно
        scale = 2.0 if min_side < 400 else 1.5
        new_size = (int(w * scale), int(h * scale))
        img_gray = img_gray.resize(new_size, Image.LANCZOS)

    # Нормализуем "силу" обработки от 0 до 1,
    # чтобы можно было управлять качеством через image_quality (0–100).
    q = max(0.0, min(1.0, float(image_quality) / 100.0))

    # Локальное выравнивание яркости: убираем медленно меняющийся фон,
    # усиливаем структуры (штрихи цифр), но оставляем естественные полутона.
    # 1) лёгкое размытие для оценки "фона"
    # радиус зависит от качества: при большем качестве сильнее выравниваем фон.
    blur_radius = 5.0 + q * 25.0  # от ~5 до ~30
    blurred = img_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 2) вычитаем фон и делаем более аккуратную нормализацию контраста
    # используем перцентили для более мягкого растяжения гистограммы
    arr_gray = np.array(img_gray).astype(np.int16)
    arr_blur = np.array(blurred).astype(np.int16)
    detail = arr_gray - arr_blur
    
    # Растягиваем гистограмму по перцентилям для более мягкой нормализации
    p2, p98 = np.percentile(detail, (2, 98))
    if p98 > p2:
        detail = (detail - p2) * (255.0 / (p98 - p2))
    detail = np.clip(detail, 0, 255).astype(np.uint8)

    img_detail = Image.fromarray(detail)

    # 3) Автоконтраст + усиление контраста: цифры становятся более чёткими.
    # cutoff делаем меньше при большем качестве (меньше "обрезаем" тени/света).
    cutoff = max(0, min(10, int(4 - 3 * q)))  # от 4 до 1
    img_detail = ImageOps.autocontrast(img_detail, cutoff=cutoff)

    # коэффициент контраста зависит от качества
    contrast_factor = 1.0 + 1.0 * q  # от 1.0 до 2.0 (немного уменьшил)
    enhancer = ImageEnhance.Contrast(img_detail)
    img_detail = enhancer.enhance(contrast_factor)

    # 4) Лёгкое повышение резкости (усиливаем края цифр, не ломая тон),
    # сила резкости также зависит от качества.
    sharpen_radius = 0.6 + 0.6 * q          # ~0.6–1.2
    sharpen_percent = int(50 + 100 * q)     # ~50–150 (немного уменьшил)
    sharpen_threshold = max(1, int(5 - 3 * q))  # ~5–2
    img_detail = img_detail.filter(
        ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold,
        )
    )

    img_final = img_detail.convert('RGB')
    return img_final


def image_to_text(
    image_path: Union[str, Path, io.BytesIO],
    min_score: float = 0.6,
    group_by_line: bool = True,
    line_threshold: float = 0.5,
    save_to_output: bool = False,
    output_name: str = None,
) -> Dict[str, List]:
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)
    
    img = _enhance_image_for_ocr(img)
    
    if save_to_output:
        # Сохраняем обработанное изображение в output
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        if isinstance(image_path, io.BytesIO):
            base_name = output_name or "enhanced_image"
        else:
            base_name = output_name or Path(image_path).stem
        
        output_path = output_dir / f"{base_name}_enhanced.jpg"
        img.save(output_path, "JPEG", quality=95)
    
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

