import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from src.ocr_config import get_ocr_kwargs


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
        "container": _build_ocr_instance(version="container"),
        "car": _build_ocr_instance(version="car"),
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
    print_variants: bool = False,
) -> Dict[str, List]:
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    if save_to_output:
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        if isinstance(image_path, io.BytesIO):
            base_name = output_name or "ocr_image"
        else:
            base_name = output_name or Path(image_path).stem
        img.save(output_dir / f"{base_name}_ocr.jpg", "JPEG", quality=95)

    ocr_version = "car" if detect == "car" else "container"
    result = _run_ocr_pass(img, ocr_version, min_score, group_by_line, line_threshold)
    current_scores = result["data"]["rec_scores"]

    if print_variants:
        print("--------------------------------")
        print(f"ocr_version: {ocr_version}")
        print(f"rec_texts: {result['data']['rec_texts']}")
        avg_score = (sum(current_scores) * 100 / len(current_scores)) if current_scores else 0.0
        print(f"rec_scores: {avg_score:.2f}%")

    return result
