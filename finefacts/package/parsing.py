"""Parse LLM responses to JSON + derive per-article metadata."""

from __future__ import annotations

import hashlib
import json
import re


def parse_json(text):
    """Extract JSON from an LLM response (tolerates markdown ```json fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[\{][\s\S]*[\]\}]", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def article_id(art, text):
    """Stable per-article ID. Use the caller's id field if present, else hash the text."""
    return art.get("id") or art.get("article_id") or \
        hashlib.blake2b(text[:200].encode(), digest_size=8).hexdigest()


def article_meta(art, text):
    """Common metadata fields written to every per-article JSON output."""
    return {
        "article_id": article_id(art, text),
        "title": art.get("title") or art.get("headline") or text.split("\n", 1)[0][:200],
        "source_url": art.get("url") or art.get("source_url", ""),
        "source_domain": art.get("base_url") or art.get("source_domain", ""),
    }
