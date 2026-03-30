import json
from pathlib import Path
from typing import Dict, List, Optional

from src.validate_container import validate_container, validate_partial_container, calc_check_digit, normalize_container_number


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

_COUNTRY_CODES: Dict[str, str] = {}

_COUNTRY_CODES_PATH = Path(__file__).resolve().parent / "car_number_countries.json"
try:
    with _COUNTRY_CODES_PATH.open("r", encoding="utf-8") as f:
        _COUNTRY_CODES = json.load(f)
except Exception:
    _COUNTRY_CODES = {}


def _get_car_number(texts: List[str]) -> str:
    for text in texts:
        if not text:
            continue
        
        cleaned = text.strip().upper()
        
        result = ''.join(c for c in cleaned if c.isalnum())
        
        if len(result) > 3:
            last_3 = result[-3:]
            last_2 = result[-2:]

            if last_3 in _COUNTRY_CODES:
                result = result[:-3]
            elif last_2 in _COUNTRY_CODES:
                result = result[:-2]
        
        if result:
            return result
    
    return ""


def _filter_by_length(texts: List[str]) -> List[str]:
    filtered = []
    for text in texts:
        original = text.strip().replace(".", "")

        parts = original.split()

        for i, part in enumerate(parts):
            if len(part) == 7:
                parts[i] = part[:-1]

        if len(parts) >= 3:
            merged = [parts[0]]
            for p in parts[1:]:
                if merged[-1].isdigit() and p.isdigit():
                    merged[-1] = merged[-1] + p
                else:
                    merged.append(p)
            parts = merged

        if len(parts) >= 2:
            part_6_idx = None
            part_1_idx = None
            
            for i, part in enumerate(parts):
                if len(part) == 6:
                    part_6_idx = i
                    break
            
            for i, part in enumerate(parts):
                if len(part) == 1:
                    part_1_idx = i
                    break
            
            if (part_6_idx is not None
                    and part_1_idx is not None
                    and part_1_idx > part_6_idx
                    and parts[part_1_idx].isdigit()):
                new_parts = []
                combined = None
                for i, part in enumerate(parts):
                    if i == part_6_idx or i == part_1_idx:
                        if combined is None:
                            combined = parts[part_6_idx] + parts[part_1_idx]
                            new_parts.append(combined)
                        continue
                    new_parts.append(part)
                parts = new_parts

        for part in parts:
            trimmed = part.replace(" ", "").replace("-", "")
            length = len(trimmed)
            if length in [4, 6, 7, 10, 11]:
                filtered.append(trimmed.upper())

    parts_4 = [item for item in filtered if len(item) == 4]
    parts_6_7 = [item for item in filtered if len(item) in [6, 7]]
    
    for part_4 in parts_4:
        for part_6_7 in parts_6_7:
            combined = part_4 + part_6_7
            if len(combined) in [10, 11] and combined not in filtered:
                filtered.append(combined)

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
        if c == "?":
            chars[i] = "2"
        if c == "C":
            chars[i] = "G"

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


def get_info(texts: List[str], detect: Optional[str] = None, scores: Optional[List[float]] = None) -> Dict[str, str]:
    container_number = None
    container_type = None
    car = None

    if detect is None:
        return {
            'number': '',
            'type': '',
            'scores_mean': 0
        }

    scores_mean = sum(scores) / len(scores) if scores else 0
    scores_mean = round(scores_mean * 100, 2)

    if detect == 'car':
        car = _get_car_number(texts)

        return {
            'car': car or '',
            'scores_mean': scores_mean
        }
    
    filtered_texts = _filter_by_length(texts)
    container_type = _get_container_type(filtered_texts)
    container_number = _get_container_number(filtered_texts)
    
    return {
        'number': container_number or '',
        'type': container_type or '',
        'scores_mean': scores_mean
    }
