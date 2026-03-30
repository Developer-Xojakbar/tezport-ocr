import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADDLE_MODELS_DIR = PROJECT_ROOT / "paddle_models"
PADDLE_DET_MODEL_DIR = PADDLE_MODELS_DIR / "PP-OCRv5_server_det"
PADDLE_REC_MODEL_DIR = PADDLE_MODELS_DIR / "PP-OCRv5_server_rec"
V3_REC_MODEL_DIR = PADDLE_MODELS_DIR / "container_rec_infer_v3"
V5_REC_MODEL_DIR = PADDLE_MODELS_DIR / "container_server_rec_infer_v5"

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



def _build_ocr_instance(version: str = "mobile") -> PaddleOCR:
    ocr_kwargs = {
        "lang": "en",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_angle_cls": True,
    }


    if version == "mobile":
        return PaddleOCR(**ocr_kwargs)

    if version == "server":
        if PADDLE_DET_MODEL_DIR.exists():
            ocr_kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
            ocr_kwargs["text_detection_model_dir"] = str(PADDLE_DET_MODEL_DIR)

        # if PADDLE_REC_MODEL_DIR.exists() and _is_latin_compatible_rec_model(PADDLE_REC_MODEL_DIR):
        if PADDLE_REC_MODEL_DIR.exists():
            ocr_kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"
            ocr_kwargs["text_recognition_model_dir"] = str(PADDLE_REC_MODEL_DIR)

        return PaddleOCR(**ocr_kwargs)

    if version == "trained_mobile_v3":
        if V3_REC_MODEL_DIR.exists():
            ocr_kwargs["text_recognition_model_name"] = "en_PP-OCRv3_mobile_rec"
            ocr_kwargs["text_recognition_model_dir"] = str(V3_REC_MODEL_DIR)
            return PaddleOCR(**ocr_kwargs)

    if version == "trained_server_v5":
        if V5_REC_MODEL_DIR.exists():
            ocr_kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"
            ocr_kwargs["text_recognition_model_dir"] = str(V5_REC_MODEL_DIR)
            return PaddleOCR(**ocr_kwargs)




_OCR_INSTANCES: Dict[str, PaddleOCR] = {
    "mobile": _build_ocr_instance(version="mobile"),
    "server": _build_ocr_instance(version="server"),
    "trained_mobile_v3": _build_ocr_instance(version="trained_mobile_v3"),
    "trained_server_v5": _build_ocr_instance(version="trained_server_v5"),
}


def _get_ocr_instance(version: str) -> PaddleOCR:
    if version not in _OCR_INSTANCES:
        _OCR_INSTANCES[version] = _build_ocr_instance(version=version)
    return _OCR_INSTANCES[version]


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



def _pick_best_channel(img_np: np.ndarray) -> np.ndarray:
    """
    Выбирает цветовой канал, в котором текст наиболее контрастен.
    Сравнивает: grayscale, LAB-L, R, G, B — по дисперсии Лапласиана
    (чем выше — тем резче/контрастнее края символов).
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]

    candidates = [gray, l_ch, img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]]

    best = gray
    best_var = 0.0
    for ch in candidates:
        v = cv2.Laplacian(ch, cv2.CV_64F).var()
        if v > best_var:
            best_var = v
            best = ch

    return best.copy()


def _enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    img_np = np.array(img)

    h, w = img_np.shape[:2]
    min_side = min(w, h)
    if min_side < 800:
        scale = 2.0 if min_side < 400 else 1.5
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    gray = _pick_best_channel(img_np)

    # Лёгкий bilateral: слегка сглаживает текстуру металла,
    # но не размывает мелкий текст.
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)

    # CLAHE — адаптивное выравнивание гистограммы по локальным патчам;
    # вытягивает контраст даже если текст почти сливается с фоном.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Вычитаем медленно меняющийся фон (освещение, градиенты на контейнере).
    blur_ksize = max(21, (max(enhanced.shape) // 8) | 1)
    bg = cv2.GaussianBlur(enhanced, (blur_ksize, blur_ksize), 0)

    diff = enhanced.astype(np.float32) - bg.astype(np.float32)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Второй проход CLAHE — после удаления фона остатки контраста максимизируем.
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe2.apply(diff)

    # Unsharp-mask: подчёркиваем края символов.
    blurred_for_sharp = cv2.GaussianBlur(result, (0, 0), 2.0)
    result = cv2.addWeighted(result, 1.5, blurred_for_sharp, -0.5, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result_rgb)


def image_to_text(
    image_path: Union[str, Path, io.BytesIO],
    detect: str = None,
    min_score: float = 0.6,
    group_by_line: bool = True,
    line_threshold: float = 0.5,
    save_to_output: bool = False,
    output_name: str = None,
    enhance_image: bool = False,
    print_variants: bool = False,
) -> Dict[str, List]:
    # Параметр оставлен для обратной совместимости.
    # Внутри всегда выполняем 4 прогона:
    # (enhance=True/False) x (ocr_version=server/mobile)
    _ = enhance_image

    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)

    if save_to_output:
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        if isinstance(image_path, io.BytesIO):
            base_name = output_name or "enhanced_image"
        else:
            base_name = output_name or Path(image_path).stem
        output_path = output_dir / f"{base_name}_enhanced.jpg"
        _enhance_image_for_ocr(img).save(output_path, "JPEG", quality=95)

    def _run_single_pass(pass_img: Image.Image, ocr_version: str) -> Dict[str, List]:
        img_array = np.array(pass_img)
        results = _get_ocr_instance(ocr_version).predict(input=img_array)

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

    runs = [
        # {"enhance": True, "ocr_version": "mobile"},
        # {"enhance": False, "ocr_version": "mobile"},
        # {"enhance": True, "ocr_version": "server"},
        # {"enhance": False, "ocr_version": "server"},
        # {"enhance": True, "ocr_version": "trained_mobile_v3"},
        # {"enhance": False, "ocr_version": "trained_mobile_v3"},
        {"enhance": True, "ocr_version": "trained_server_v5"},
        # {"enhance": False, "ocr_version": "trained_server_v5"},
    ]

    if (detect == 'car'):
        runs = [
            {"enhance": True, "ocr_version": "server"},
        ]

    best_result: Optional[Dict[str, List]] = None
    best_key: Tuple[float, float, int] = (-1.0, -1.0, -1)

    for run in runs:
        pass_img = _enhance_image_for_ocr(img) if run["enhance"] else img
        current_result = _run_single_pass(pass_img, run["ocr_version"])
        current_scores = current_result["data"]["rec_scores"]

        if print_variants:
            print("--------------------------------")
            print(f"ocr_version: {run['ocr_version']}, enhance: {run['enhance']}")
            print(f"rec_texts: {current_result['data']['rec_texts']}")
            avg_score = (sum(current_scores) * 100 / len(current_scores)) if current_scores else 0.0
            print(f"rec_scores: {avg_score:.2f}%")

        current_key = (
            float(sum(current_scores)),
            float(max(current_scores) if current_scores else 0.0),
            len(current_scores),
        )
        if current_key > best_key:
            best_key = current_key
            best_result = current_result

    return best_result or {
        "data": {
            "rec_texts": [],
            "rec_scores": [],
        },
        "texts": [],
    }

