import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from validate_container import validate_container, validate_partial_container, calc_check_digit, normalize_container_number


_CONTAINER_TYPE_MAP: Dict[str, str] = {}

_CONTAINER_TYPE_PATH = Path(__file__).resolve().parent / "container_type.json"
try:
    with _CONTAINER_TYPE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
        _CONTAINER_TYPE_MAP = {
            str(k): (v if isinstance(v, str) else v.get("display", ""))
            for k, v in data.items()
        }
except Exception:
    _CONTAINER_TYPE_MAP = {}


def _filter_by_length(texts: List[str]) -> List[str]:
    filtered = []
    for text in texts:
        trimmed = text.strip().replace(" ", "").replace("-", "")
        length = len(trimmed)
        if length in [4, 10, 11]:
            filtered.append(trimmed.upper())
    return filtered


def _normalize_type_code(raw: str) -> Optional[str]:
    if not raw:
        return None

    code = raw
    if len(code) != 4:
        return None

    chars = list(code)

    for i, c in enumerate(chars):
        if c == "O":
            chars[i] = "0"

    if chars[2] == "6":
        chars[2] = "G"

    if chars[3] in ("G"):
        chars[3] = "6"

    return "".join(chars)


def _get_container_type(texts: List[str]) -> str:
    if not _CONTAINER_TYPE_MAP:
        return ""

    for text in texts:
        if len(text) != 4:
            continue

        code = _normalize_type_code(text)
        if not code:
            continue

        if code in _CONTAINER_TYPE_MAP:
            return _CONTAINER_TYPE_MAP[code]

    return ""


def _get_container_number(texts: List[str]) -> str:
    normalized_texts = [normalize_container_number(text) for text in texts]

    for text in normalized_texts:
        if len(text) == 11:
            if validate_container(text):
                return text

    for text in normalized_texts:
        if len(text) == 10:
            if validate_partial_container(text):
                check_digit = calc_check_digit(text)
                return f"{text}{check_digit}"

    return ""


def get_container_info(texts: List[str]) -> Dict[str, str]:
    container_number = None
    container_type = None
    
    filtered_texts = _filter_by_length(texts)
    container_type = _get_container_type(filtered_texts)
    container_number = _get_container_number(filtered_texts)
    
    return {
        'number': container_number or '',
        'type': container_type or ''
    }
