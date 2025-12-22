import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def image_to_text(
    image_path: Union[str, Path, io.BytesIO],
    ocr_instance: Optional[PaddleOCR] = None,
    lang: str = "en",
) -> Dict[str, List]:
    if ocr_instance is None:
        ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    # Если это BytesIO, конвертируем в numpy array для PaddleOCR
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
        img_array = np.array(img)
        results = ocr_instance.predict(input=img_array)
    else:
        # Если это путь к файлу, используем как обычно
        image_path_str = str(image_path)
        results = ocr_instance.predict(input=image_path_str)

    rec_texts: List[str] = []
    rec_scores: List[float] = []

    if results:
        for res in results:
            if isinstance(res, dict):
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
            else:
                texts = getattr(res, "rec_texts", None) or []
                scores = getattr(res, "rec_scores", None) or []

            if texts:
                rec_texts.extend(texts)
                if scores:
                    rec_scores.extend(scores)
                else:
                    rec_scores.extend([0.0] * len(texts))

    return {
        "data": {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        },
        "texts": rec_texts,
    }

