from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADDLE_MODELS_DIR = PROJECT_ROOT / "paddle_models"
PADDLE_DET_MODEL_DIR = PADDLE_MODELS_DIR / "PP-OCRv5_server_det"
PADDLE_REC_MODEL_DIR = PADDLE_MODELS_DIR / "PP-OCRv5_server_rec"
V3_REC_MODEL_DIR = PADDLE_MODELS_DIR / "container_rec_infer_v3"
V5_REC_MODEL_DIR = PADDLE_MODELS_DIR / "container_server_rec_infer_v5"

OCR_VERSIONS: List[str] = [
    "mobile",
    "server",
    "trained_mobile_v3",
    "trained_server_v5",
]


def _get_ocr_device() -> Optional[str]:
    return "gpu:0"


def get_ocr_kwargs(version: str) -> Dict:
    ocr_kwargs = {
        "lang": "en",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "enable_mkldnn": False,
    }

    device = _get_ocr_device()
    if device is not None:
        ocr_kwargs["device"] = device

    if version == "mobile":
        return ocr_kwargs

    if version == "server":
        if PADDLE_DET_MODEL_DIR.exists():
            ocr_kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
            ocr_kwargs["text_detection_model_dir"] = str(PADDLE_DET_MODEL_DIR)
        if PADDLE_REC_MODEL_DIR.exists():
            ocr_kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"
            ocr_kwargs["text_recognition_model_dir"] = str(PADDLE_REC_MODEL_DIR)
        return ocr_kwargs

    if version == "trained_mobile_v3" and V3_REC_MODEL_DIR.exists():
        ocr_kwargs["text_recognition_model_name"] = "en_PP-OCRv3_mobile_rec"
        ocr_kwargs["text_recognition_model_dir"] = str(V3_REC_MODEL_DIR)
        return ocr_kwargs

    if version == "trained_server_v5" and V5_REC_MODEL_DIR.exists():
        ocr_kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"
        ocr_kwargs["text_recognition_model_dir"] = str(V5_REC_MODEL_DIR)
        return ocr_kwargs

    raise ValueError(f"Unsupported or misconfigured OCR version: {version}")
