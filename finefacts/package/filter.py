"""Corpus pre-filtering — decide which articles to process before extraction.

`ff.extract(filter=)` takes any `Callable[[dict], bool]`. This module has two
ready-made factories: `keyword_filter` (substring, free) and `llm_filter`
(LLM classifier, cheap). `compose_filters` combines them with AND / OR.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable


# ── keyword filter — pure Python, no API cost ─────────────────────────────


def keyword_filter(
    keywords: Iterable[str],
    *,
    field: str = "text",
    match: str = "any",
    case_sensitive: bool = False,
) -> Callable[[dict], bool]:
    """Return a filter that keeps articles whose `field` contains keywords.

    Args:
        keywords:        iterable of substrings to search for.
        field:           article field to search. Defaults to "text"; try
                         "title" for headline-only matching.
        match:           "any" (default) — keep if ANY keyword matches.
                         "all" — keep only when ALL keywords match.
        case_sensitive:  substring compare mode.

    Example:
        f = ff.keyword_filter(["Russia", "Ukraine"], match="any")
        ff.extract(corpus, prompt=..., output=..., filter=f)
    """
    kws = [str(k) for k in keywords]
    if not kws:
        raise ValueError("keyword_filter requires at least one keyword")
    if match not in ("any", "all"):
        raise ValueError(f"match must be 'any' or 'all'; got {match!r}")
    if not case_sensitive:
        kws = [k.lower() for k in kws]

    def check(art: dict) -> bool:
        text = art.get(field, "")
        if not isinstance(text, str) or not text:
            return False
        if not case_sensitive:
            text = text.lower()
        if match == "all":
            return all(k in text for k in kws)
        return any(k in text for k in kws)

    check.__name__ = f"keyword_filter({', '.join(kws[:3])}{'...' if len(kws) > 3 else ''})"
    return check


# ── LLM classifier filter — cheap gate before expensive extraction ──


def llm_filter(
    classifier_prompt: str,
    *,
    model: str = "claude-haiku-4-5",
    positive_pattern: str = r"\byes\b",
    max_chars: int = 4000,
    cache_dir=None,
    tracker=None,
) -> Callable[[dict], bool]:
    """Return a filter that uses a cheap LLM to classify each article.

    The classifier prompt should elicit a short YES/NO-ish response; the
    `positive_pattern` regex is matched (case-insensitive) against that
    response to decide keep vs skip.

    Args:
        classifier_prompt: system prompt for the classifier LLM. Should be
                           short and direct — e.g. "Is this article about the
                           Russia-Ukraine war? Respond YES or NO."
        model:             cheap model (haiku, flash, mini) — you're paying
                           for this on EVERY article, so pick something small.
        positive_pattern:  regex; article kept when this matches the response.
                           Default `\\byes\\b` (case-insensitive).
        max_chars:         truncate article to this many chars before sending
                           to the classifier. Classification rarely needs the
                           whole article and truncating cuts cost.
        cache_dir:         optional persistent cache; identical (article,
                           classifier_prompt, model) tuples reuse prior
                           verdicts across runs. Free resume on rerun.
        tracker:           optional `CostTracker` — classifier calls count
                           against the same `max_spend` cap as extraction.

    Example:
        classifier = ff.llm_filter(
            "Is this article primarily about the Russia-Ukraine war? "
            "Respond YES or NO.",
            model="claude-haiku-4-5",
        )
        ff.extract(
            corpus=all_wsm_articles,
            prompt=extract_prompt,
            output="./out/",
            filter=classifier,
        )
    """
    if not isinstance(classifier_prompt, str) or not classifier_prompt.strip():
        raise ValueError("classifier_prompt must be a non-empty string")
    pattern = re.compile(positive_pattern, re.IGNORECASE | re.MULTILINE)

    # Local imports keep this module dep-free at import time.
    from . import llms
    from .cache import cached_call

    def check(art: dict) -> bool:
        text = (art.get("text") or art.get("body") or art.get("content") or "")
        text = text[:max_chars]
        title = art.get("title") or art.get("headline") or ""
        user_msg = f"Title: {title}\n\n{text}"
        if not text.strip():
            return False  # nothing to classify → skip
        try:
            if cache_dir:
                response = cached_call(
                    model, classifier_prompt, user_msg, cache_dir,
                    lambda: llms.call_llm(model, classifier_prompt, user_msg,
                                           tracker=tracker),
                )
            else:
                response = llms.call_llm(model, classifier_prompt, user_msg,
                                         tracker=tracker)
        except Exception:
            # Classifier failure → conservatively KEEP the article.
            # (Better to over-include than silently drop data.)
            return True
        return bool(pattern.search(response or ""))

    check.__name__ = f"llm_filter(model={model!r})"
    return check


# ── composition — combine multiple filters ────────────────────────────────


def compose_filters(*filters, mode: str = "all") -> Callable[[dict], bool]:
    """Combine any mix of filter callables into a single predicate.

    Args:
        *filters:  one or more `Callable[[dict], bool]`. Mix built-ins like
                   `keyword_filter` and `llm_filter` freely with your own
                   Python functions.
        mode:      "all" (default, AND semantics) — every filter must return
                   True to keep the article.
                   "any" (OR semantics) — the article is kept if ANY filter
                   returns True. Short-circuits on the first match.

    Example:
        must_be_state_media = lambda art: art.get("source_domain") in STATE_DOMAINS
        must_mention_topic  = ff.keyword_filter(["Russia", "Ukraine"])
        cheap_classifier    = ff.llm_filter("Is this news? YES/NO.")

        # An article is kept only if ALL three conditions hold.
        combined = ff.compose_filters(
            must_be_state_media,
            must_mention_topic,
            cheap_classifier,
            mode="all",
        )
        ff.extract(corpus=..., filter=combined, ...)
    """
    if not filters:
        raise ValueError("compose_filters requires at least one filter")
    for i, f in enumerate(filters):
        if not callable(f):
            raise ValueError(
                f"compose_filters[{i}] must be callable; got {type(f).__name__}"
            )
    if mode not in ("all", "any"):
        raise ValueError(f"mode must be 'all' or 'any'; got {mode!r}")

    def check(art: dict) -> bool:
        if mode == "all":
            return all(f(art) for f in filters)
        return any(f(art) for f in filters)

    names = [getattr(f, "__name__", "filter") for f in filters]
    check.__name__ = f"compose_filters({mode}: {', '.join(names[:3])}{'...' if len(names) > 3 else ''})"
    return check
