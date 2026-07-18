"""Cross-lingual extraction — `output_language=` and `translate_first=`.

`output_language="en"` appends a language directive to the system prompt
(1 call/article). `translate_first=True` adds a pre-translation call so the
translation is saved as an artifact (1 + N calls/article). Identifiers are
ISO 639-1 codes or English names.
"""

from __future__ import annotations


# Curated ISO 639-1 → English-name table. Not exhaustive; unknown codes
# pass through.
_LANG_NAMES = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
    "zh-cn": "Simplified Chinese", "zh-tw": "Traditional Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
    "ar": "Arabic", "pt": "Portuguese", "it": "Italian", "hi": "Hindi",
    "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "uk": "Ukrainian",
    "he": "Hebrew", "fa": "Persian", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "da": "Danish", "fi": "Finnish",
    "no": "Norwegian", "bg": "Bulgarian", "sk": "Slovak", "ms": "Malay",
}


def canonical_language_name(lang: str | None) -> str | None:
    """Normalize 'en' / 'english' / 'EN' → 'English' for display."""
    if lang is None:
        return None
    s = str(lang).strip().lower()
    if s in _LANG_NAMES:
        return _LANG_NAMES[s]
    for code, name in _LANG_NAMES.items():
        if name.lower() == s:
            return name
    return s.title()  # unknown — pass user's string capitalized


def language_directive(target_lang: str) -> str:
    """Instruction appended to system prompts for direct cross-lingual mode."""
    name = canonical_language_name(target_lang)
    return (
        f"IMPORTANT — language: The article you are given may be in any "
        f"language. Always produce your output in {name}. Translate any "
        f"non-{name} content from the article as needed during extraction. "
        f"Preserve names, numbers, dates, and quoted material exactly."
    )


def translation_system_prompt(target_lang: str) -> str:
    """Stage-1 system prompt for translate-then-extract mode."""
    name = canonical_language_name(target_lang)
    return (
        f"You are a high-fidelity translator. Translate the article below to "
        f"{name}. Preserve names, numbers, dates, quoted material, and the "
        f"original structure (paragraphs, lists, headings). If the article "
        f"is already in {name}, return it unchanged. Output ONLY the "
        f"translated article text — no commentary, no framing."
    )


def wrap_prompts_with_language(prompts, target_lang: str):
    """Append the language directive to each system prompt.

    Accepts all three prompt shapes (str / list[str] / dict[str, str]) and
    returns the same shape with each system prompt augmented.
    """
    directive = "\n\n" + language_directive(target_lang)
    if isinstance(prompts, str):
        return prompts + directive
    if isinstance(prompts, list):
        return [p + directive for p in prompts]
    if isinstance(prompts, dict):
        return {k: v + directive for k, v in prompts.items()}
    raise TypeError(
        f"prompts must be str / list[str] / dict[str, str]; "
        f"got {type(prompts).__name__}"
    )
