from __future__ import annotations

import re
from typing import Final


ISO_VALUES: Final[dict[str, int]] = {
    "A": 10,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 23,
    "M": 24,
    "N": 25,
    "O": 26,
    "P": 27,
    "Q": 28,
    "R": 29,
    "S": 30,
    "T": 31,
    "U": 32,
    "V": 34,
    "W": 35,
    "X": 36,
    "Y": 37,
    "Z": 38,
}

CONTAINER_FULL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{4}\d{7}$")


def normalize_container_number(value: str) -> str:
    if not value:
        return ""

    code = value
    chars = list(code)

    for i in range(min(4, len(chars))):
        if chars[i] == "0":
            chars[i] = "O"
        elif chars[i] == "6":
            chars[i] = "G"

    for i in range(4, len(chars)):
        if chars[i] == "O":
            chars[i] = "0"
        elif chars[i] == "G":
            chars[i] = "6"

    return "".join(chars)


def calc_check_digit(code: str) -> int:
    base = code[:10].upper()
    total = 0

    for i, c in enumerate(base):
        if c.isdigit():
            value = int(c)
        else:
            value = ISO_VALUES.get(c, 0)
        total += value * (2**i)

    mod = total % 11
    return 0 if mod == 10 else mod


def validate_container(num: str, without_iso_check: bool = False) -> bool:
    code = num

    if not CONTAINER_FULL_RE.match(code):
        return False
    if without_iso_check:
        return True

    check_digit = calc_check_digit(code)
    provided = int(code[10])
    return check_digit == provided


def _extend_container_number(value: str) -> str:
    length_difference = 11 - len(value)
    if length_difference <= 0:
        return value

    final_value = value
    digits = 0
    letters = 0

    for ch in value:
        code = ord(ch)
        if 48 <= code <= 57:
            digits += 1
        if 65 <= code <= 90:
            letters += 1

    digits_diff = 7 - digits
    letters_diff = 4 - letters

    for _ in range(letters_diff):
        if digits_diff == 7:
            final_value = final_value + "Z"
        else:
            final_value = "Z" + final_value

    for _ in range(digits_diff):
        final_value += "1"

    return final_value


def validate_partial_container(value: str) -> bool:
    if not value:
        return False

    code = value

    if len(code) == 11:
        return validate_container(code)

    extended = _extend_container_number(code)
    return bool(CONTAINER_FULL_RE.match(extended))
