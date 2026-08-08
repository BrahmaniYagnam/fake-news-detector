import re
from typing import Iterable


def clean_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_keywords(tokens: Iterable[str], limit: int = 10) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)
        if len(keywords) >= limit:
            break
    return keywords
