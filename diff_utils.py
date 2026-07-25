"""
Text change detection utilities.

Uses Levenshtein ratio with a normalized character map to avoid
false triggers from OCR misreads of Polish diacritics.
(e.g. '$' read as 'ś', '0' as 'o', '|' as 'l')
"""
import re
from Levenshtein import ratio as lev_ratio

# ─── OCR misread normalisation map ─────────────────────────────────────────
# Maps common ASCII misreads → correct Polish/Latin characters
CHAR_NORM_MAP: dict[str, str] = {
    # Polish diacritics
    "$": "ś",
    "§": "ś",
    "^": "ą",
    "~": "ż",
    "#": "ń",
    "@": "ę",
    "&": "ź",
    # Generic confusions
    "|": "l",
    "!": "i",
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "8": "b",
}

# Similarity below this threshold → "significant change"
# 0.70 → texts must differ in at least 30% of characters
_CHANGE_THRESHOLD = 0.70

# Minimum non-empty length to trigger TTS (avoids single-char noise)
_MIN_TEXT_LENGTH = 3


def _normalize(text: str) -> str:
    """Lowercase, strip, apply misread map, collapse whitespace."""
    text = text.lower().strip()
    for src, dst in CHAR_NORM_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text


def has_significant_change(old_text: str, new_text: str) -> bool:
    """
    Return True when new_text differs from old_text enough to warrant
    a new TTS generation.

    Rules:
    - Both empty → no change
    - Old empty, new non-empty → change (new subtitle appeared)
    - New empty (OCR returned nothing) → no change (transient blank frame)
    - New text too short → no change (noise)
    - Levenshtein similarity below threshold → change
    """
    old_n = _normalize(old_text)
    new_n = _normalize(new_text)

    if not old_n and not new_n:
        return False

    if not new_n:
        return False  # Don't trigger on blank OCR frame

    if len(new_n) < _MIN_TEXT_LENGTH:
        return False  # Noise / single character

    if not old_n:
        return True  # Fresh subtitle after silence

    similarity = lev_ratio(old_n, new_n)
    return similarity < _CHANGE_THRESHOLD
