"""
language.py: decide the reply language IN CODE, per message.

Why code and not only the prompt: the model sees Arabic memories, an Arabic
summary and Arabic documents, so a prompt rule like "reply in the user's
language" is easy for it to get wrong (e.g. "hi" -> Arabic reply). Here we
detect the language of the CURRENT message and tell the model exactly which
language to use. Unsupported languages are answered without calling the model.

Rules:
  - Mostly Arabic script            -> "ar"
      (but Persian/Urdu letters such as پ چ ژ گ ک ی ٹ ڈ ڑ ں ے -> "other")
  - Mostly Latin script             -> "en", unless clearly another language
      (French, Spanish, German, ...) -> "other". Short or unclear Latin text
      ("hi", "ok", "doc-3e47c5") counts as English, to avoid false refusals.
  - Other scripts (Cyrillic, Chinese, Hindi, Hebrew, ...) -> "other"
  - No letters at all ("2**10", "?")  -> None: keep the previous language

Optional, better detection for Latin text:
  pip install lingua-language-detector     (~100 MB; works on short text)
Without it, all Latin-script text is treated as English.
"""

import unicodedata
from typing import Optional

ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
PERSIAN_URDU_LETTERS = set("پچژگکیٹڈڑںےہھ")

REFUSAL = (
    "عذراً، أفهم اللغتين العربية والإنجليزية فقط.\n"
    "Sorry, I only understand Arabic and English."
)
LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}

try:  # optional
    from lingua import Language, LanguageDetectorBuilder

    _OTHER_LATIN = [Language.FRENCH, Language.SPANISH, Language.GERMAN,
                    Language.ITALIAN, Language.PORTUGUESE, Language.TURKISH,
                    Language.DUTCH, Language.INDONESIAN, Language.MALAY,
                    Language.TAGALOG, Language.SWAHILI, Language.POLISH]
    _detector = (LanguageDetectorBuilder
                 .from_languages(Language.ENGLISH, *_OTHER_LATIN)
                 .with_minimum_relative_distance(0.25)  # unsure -> None
                 .build())
except ImportError:
    _detector = None


def _is_arabic(ch: str) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a, b in ARABIC_RANGES)


def _is_latin(ch: str) -> bool:
    return ("a" <= ch.lower() <= "z") or (0x00C0 <= ord(ch) <= 0x024F)


def detect_language(text: str) -> Optional[str]:
    """Return "ar", "en", "other", or None (no letters: keep previous)."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return None

    arabic = [c for c in letters if _is_arabic(c)]
    latin = [c for c in letters if _is_latin(c)]
    other = len(letters) - len(arabic) - len(latin)

    if other > max(len(arabic), len(latin)):
        return "other"                                   # Cyrillic, CJK, Devanagari...

    if len(arabic) >= len(latin):                        # Arabic script wins ties
        pu = sum(1 for c in arabic if c in PERSIAN_URDU_LETTERS)
        if pu >= 2 and pu / len(arabic) >= 0.05:
            return "other"                               # Persian / Urdu
        return "ar"

    if _detector is not None:                            # Latin script
        lang = _detector.detect_language_of(text)
        if lang is not None and lang != Language.ENGLISH:
            return "other"
    return "en"


def language_instruction(lang: str) -> str:
    name = LANGUAGE_NAMES[lang]
    return (f"Reply in {name} only, because the user's current message is in {name}. "
            "This overrides the language of memories, the chat summary, earlier "
            "turns and documents. Keep quotes from documents in their original wording.")
