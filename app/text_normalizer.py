"""Vietnamese display/spoken-text normalization for predictable TTS."""

from __future__ import annotations

import re

_DIGITS = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"}
_UNITS = {
    "m²": "mét vuông", "m2": "mét vuông", "m³": "mét khối", "m3": "mét khối",
    "km/h": "ki lô mét một giờ", "km": "ki lô mét", "cm": "xen ti mét",
    "mm": "mi li mét", "kg": "ki lô gam", "g": "gam", "%": "phần trăm",
}


def vietnamese_integer(value: int) -> str:
    if value < 0:
        return "âm " + vietnamese_integer(-value)
    if value < 10:
        return _DIGITS[str(value)]
    if value < 20:
        tail = value % 10
        return "mười" + (" " + ("lăm" if tail == 5 else _DIGITS[str(tail)]) if tail else "")
    if value < 100:
        tens, tail = divmod(value, 10)
        ending = "" if tail == 0 else " " + ("mốt" if tail == 1 else "lăm" if tail == 5 else _DIGITS[str(tail)])
        return f"{_DIGITS[str(tens)]} mươi{ending}"
    if value < 1000:
        hundreds, tail = divmod(value, 100)
        if not tail:
            return f"{_DIGITS[str(hundreds)]} trăm"
        bridge = " lẻ " if tail < 10 else " "
        return f"{_DIGITS[str(hundreds)]} trăm{bridge}{vietnamese_integer(tail)}"
    if value < 1_000_000:
        thousands, tail = divmod(value, 1000)
        return vietnamese_integer(thousands) + " nghìn" + (" " + vietnamese_integer(tail) if tail else "")
    return " ".join(_DIGITS[digit] for digit in str(value))


def normalize_spoken_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    units = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))

    def replace_measure(match: re.Match[str]) -> str:
        number, unit = match.group(1), match.group(2)
        spoken = vietnamese_integer(int(number)) if number.isdigit() and len(number) <= 6 else " ".join(_DIGITS.get(c, c) for c in number)
        return f"{spoken} {_UNITS[unit.lower()]}"

    normalized = re.sub(rf"(?<!\w)(\d+)\s*({units})(?!\w)", replace_measure, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()
