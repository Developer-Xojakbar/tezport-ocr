from pathlib import Path
from typing import Dict, List, Optional

from src.settings import CAR_DET, CAR_REC, CONTAINER_DET, CONTAINER_REC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADDLE_MODELS_DIR = PROJECT_ROOT / "paddle_models"
TEXTLINE_ORI_MODEL_DIR = PADDLE_MODELS_DIR / "PP-LCNet_x1_0_textline_ori"

OCR_VERSIONS: List[str] = ["container", "car"]

PADDLE_MODEL_NAMES = {
    "PP-OCRv6_medium_det": "PP-OCRv6_medium_det",
    "PP-OCRv5_server_det": "PP-OCRv5_server_det",
    "PP-OCRv6_medium_rec": "PP-OCRv6_medium_rec",
    "PP-OCRv5_server_rec": "PP-OCRv5_server_rec",
    "container_rec_v6_infer": "PP-OCRv6_medium_rec",
    "container_server_rec_infer_v5": "PP-OCRv5_server_rec",
    "container_rec_infer_v3": "en_PP-OCRv3_mobile_rec",
}


def _get_ocr_device() -> Optional[str]:
    return "gpu:0"


def _has_infer_model(model_dir: Path) -> bool:
    return (model_dir / "inference.pdiparams").exists()


def _require_infer_model(folder: str) -> str:
    model_dir = PADDLE_MODELS_DIR / folder
    if not _has_infer_model(model_dir):
        raise ValueError(
            f"Local model '{folder}' not found: {model_dir}. "
            f"Run: python scripts/download_paddle_models.py"
        )
    return str(model_dir.resolve())


def _paddle_name(folder: str) -> str:
    return PADDLE_MODEL_NAMES.get(folder, folder)


def _apply_local_textline(ocr_kwargs: Dict) -> None:
    if _has_infer_model(TEXTLINE_ORI_MODEL_DIR):
        ocr_kwargs["textline_orientation_model_name"] = "PP-LCNet_x1_0_textline_ori"
        ocr_kwargs["textline_orientation_model_dir"] = str(TEXTLINE_ORI_MODEL_DIR.resolve())
        ocr_kwargs["use_textline_orientation"] = True
    else:
        ocr_kwargs["use_textline_orientation"] = False


def _base_ocr_kwargs() -> Dict:
    ocr_kwargs = {
        "lang": "en",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "enable_mkldnn": False,
    }
    device = _get_ocr_device()
    if device is not None:
        ocr_kwargs["device"] = device
    _apply_local_textline(ocr_kwargs)
    return ocr_kwargs


def get_ocr_kwargs(version: str) -> Dict:
    if version == "container":
        det_folder, rec_folder = CONTAINER_DET, CONTAINER_REC
    elif version == "car":
        det_folder, rec_folder = CAR_DET, CAR_REC
    else:
        raise ValueError(f"Unsupported OCR version: {version}")

    ocr_kwargs = _base_ocr_kwargs()
    ocr_kwargs["text_detection_model_name"] = _paddle_name(det_folder)
    ocr_kwargs["text_detection_model_dir"] = _require_infer_model(det_folder)
    ocr_kwargs["text_recognition_model_name"] = _paddle_name(rec_folder)
    ocr_kwargs["text_recognition_model_dir"] = _require_infer_model(rec_folder)
    return ocr_kwargs
