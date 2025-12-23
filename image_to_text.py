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
    min_score: float = 0.0,
) -> Dict[str, List]:
    if ocr_instance is None:
        ocr_instance = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)
        img = Image.open(image_path)
        img_array = np.array(img)
        results = ocr_instance.predict(input=img_array)
    else:
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
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    
                    score = 0.0
                    if scores and i < len(scores):
                        score = scores[i]
                    
                    if score >= min_score:
                        rec_texts.append(text)
                        rec_scores.append(score)

    return {
        "data": {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        },
        "texts": rec_texts,
    }

