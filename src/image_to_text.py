import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.ocr_config import (
    PADDLE_DET_MODEL_DIR,
    PADDLE_MODELS_DIR,
    PADDLE_REC_MODEL_DIR,
    V3_REC_MODEL_DIR,
    V5_REC_MODEL_DIR,
    get_ocr_kwargs,
)


def _use_ocr_subprocess() -> bool:
    """На Windows PyTorch и Paddle GPU конфликтуют — OCR в отдельном процессе."""
    if sys.platform != "win32":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


USE_OCR_SUBPROCESS = _use_ocr_subprocess()


def _cleanup_paddle_modules() -> None:
    for key in list(sys.modules.keys()):
        if key == "paddle" or key.startswith("paddle."):
            del sys.modules[key]


def _check_gpu_available() -> bool:
    try:
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            return False
        try:
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                paddle.device.set_device("gpu:0")
                return True
            return False
        except Exception:
            return False
    except ImportError:
        return False
    except Exception:
        _cleanup_paddle_modules()
        return False


if USE_OCR_SUBPROCESS:
    USE_GPU = False
    _ocr_subprocess_client = None
else:
    from paddleocr import PaddleOCR

    USE_GPU = _check_gpu_available()
    if USE_GPU:
        print("✅ GPU обнаружен! PaddleOCR будет использовать GPU для ускорения.")
    else:
        print("ℹ️ GPU не обнаружен или недоступен. Используется CPU.")


def _get_ocr_subprocess_client():
    global _ocr_subprocess_client
    if _ocr_subprocess_client is None:
        from src.ocr_subprocess import get_ocr_subprocess_client
        _ocr_subprocess_client = get_ocr_subprocess_client()
    return _ocr_subprocess_client


def _build_ocr_instance(version: str = "mobile"):
    from paddleocr import PaddleOCR
    return PaddleOCR(**get_ocr_kwargs(version))


_OCR_INSTANCES: Dict[str, object] = {}


def _ensure_ocr_instance(version: str):
    if version not in _OCR_INSTANCES:
        _OCR_INSTANCES[version] = _build_ocr_instance(version=version)
    return _OCR_INSTANCES[version]


if not USE_OCR_SUBPROCESS:
    _OCR_INSTANCES = {
        "mobile": _build_ocr_instance(version="mobile"),
        "server": _build_ocr_instance(version="server"),
        "trained_mobile_v3": _build_ocr_instance(version="trained_mobile_v3"),
        "trained_server_v5": _build_ocr_instance(version="trained_server_v5"),
    }


def _get_ocr_instance(version: str):
    if USE_OCR_SUBPROCESS:
        return _get_ocr_subprocess_client()
    return _ensure_ocr_instance(version)


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
                "text": text,
                "score": score,
                "y": avg_y,
                "height": height,
                "x": min_x,
            })
        except (IndexError, TypeError, ValueError, AttributeError):
            continue

    if not text_items:
        return texts, scores

    text_items.sort(key=lambda x: x["y"])

    grouped_texts = []
    grouped_scores = []
    current_line = []
    current_line_y = None
    current_line_height = 0

    for item in text_items:
        if current_line_y is None:
            current_line = [item]
            current_line_y = item["y"]
            current_line_height = item["height"]
        else:
            y_diff = abs(item["y"] - current_line_y)
            threshold = max(current_line_height, item["height"]) * line_threshold

            if y_diff <= threshold:
                current_line.append(item)
                current_line_height = max(current_line_height, item["height"])
            else:
                if current_line:
                    current_line.sort(key=lambda x: x["x"])
                    combined_text = " ".join([item["text"] for item in current_line])
                    avg_score = sum([item["score"] for item in current_line]) / len(current_line)
                    grouped_texts.append(combined_text)
                    grouped_scores.append(avg_score)

                current_line = [item]
                current_line_y = item["y"]
                current_line_height = item["height"]

    if current_line:
        current_line.sort(key=lambda x: x["x"])
        combined_text = " ".join([item["text"] for item in current_line])
        avg_score = sum([item["score"] for item in current_line]) / len(current_line)
        grouped_texts.append(combined_text)
        grouped_scores.append(avg_score)

    return grouped_texts, grouped_scores


def _pick_best_channel(img_np: np.ndarray) -> np.ndarray:
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
    if img.mode != "RGB":
        img = img.convert("RGB")

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
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur_ksize = max(21, (max(enhanced.shape) // 8) | 1)
    bg = cv2.GaussianBlur(enhanced, (blur_ksize, blur_ksize), 0)

    diff = enhanced.astype(np.float32) - bg.astype(np.float32)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe2.apply(diff)

    blurred_for_sharp = cv2.GaussianBlur(result, (0, 0), 2.0)
    result = cv2.addWeighted(result, 1.5, blurred_for_sharp, -0.5, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result_rgb)


def _run_ocr_pass(
    pass_img: Image.Image,
    ocr_version: str,
    min_score: float,
    group_by_line: bool,
    line_threshold: float,
) -> Dict[str, List]:
    img_array = np.array(pass_img)
    ocr = _get_ocr_instance(ocr_version)

    if USE_OCR_SUBPROCESS:
        parsed = ocr.predict(ocr_version, img_array, min_score)
        rec_texts = parsed["rec_texts"]
        rec_scores = parsed["rec_scores"]
        rec_bboxes = parsed["rec_bboxes"]
    else:
        results = ocr.predict(input=img_array)
        rec_texts = []
        rec_scores = []
        rec_bboxes = []

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

    runs = [
        {"enhance": True, "ocr_version": "trained_server_v5"},
    ]

    if detect == "car":
        runs = [
            {"enhance": True, "ocr_version": "server"},
        ]

    best_result: Optional[Dict[str, List]] = None
    best_key: Tuple[float, float, int] = (-1.0, -1.0, -1)

    for run in runs:
        pass_img = _enhance_image_for_ocr(img) if run["enhance"] else img
        current_result = _run_ocr_pass(
            pass_img,
            run["ocr_version"],
            min_score,
            group_by_line,
            line_threshold,
        )
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
