"""Отдельный процесс PaddleOCR (Windows: без PyTorch в том же процессе)."""
import base64
import io
import json
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from paddleocr import PaddleOCR

from src.ocr_config import OCR_VERSIONS, _get_ocr_device, get_ocr_kwargs


def _build_instances() -> Dict[str, PaddleOCR]:
    instances: Dict[str, PaddleOCR] = {}
    for version in OCR_VERSIONS:
        try:
            instances[version] = PaddleOCR(**get_ocr_kwargs(version))
        except ValueError:
            continue
    return instances


def _parse_results(results: Any, min_score: float) -> Dict[str, List]:
    rec_texts: List[str] = []
    rec_scores: List[float] = []
    rec_bboxes: List[List[List[int]]] = []

    if not results:
        return {"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_bboxes": rec_bboxes}

    for res in results:
        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            bboxes = res.get("dt_polys", []) or res.get("boxes", [])
        else:
            texts = getattr(res, "rec_texts", None) or []
            scores = getattr(res, "rec_scores", None) or []
            bboxes = getattr(res, "dt_polys", None) or getattr(res, "boxes", None) or []

        for i, text in enumerate(texts):
            if not text:
                continue

            score = float(scores[i]) if scores and i < len(scores) else 0.0
            bbox = bboxes[i] if bboxes and i < len(bboxes) else []

            if score >= min_score:
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                rec_texts.append(text)
                rec_scores.append(score)
                rec_bboxes.append(bbox)

    return {"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_bboxes": rec_bboxes}


def _decode_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return np.array(img.convert("RGB"))


def main() -> None:
    device = _get_ocr_device() or "gpu"
    use_gpu = device.startswith("gpu")
    instances = _build_instances()
    print(json.dumps({"ready": True, "gpu": use_gpu, "device": device, "versions": list(instances.keys())}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            version = request["version"]
            min_score = float(request.get("min_score", 0.6))
            image_b64 = request["image_b64"]

            if version not in instances:
                raise ValueError(f"Unknown OCR version: {version}")

            img_array = _decode_image(image_b64)
            results = instances[version].predict(input=img_array)
            payload = _parse_results(results, min_score)
            payload["ok"] = True
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}

        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
