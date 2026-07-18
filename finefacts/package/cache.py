"""Persistent LLM-call cache.

Keyed by sha256(model + system + user [+ schema]). One JSON file per call.

Used by `finefacts.optimize` to avoid re-extracting the same article across
optimization iterations, and by `extractor.extract_paid*` for free resume on
identical (model, prompt, input [, schema]) tuples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable


def _key(model: str, system: str, user: str, schema_repr: str = "") -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    if schema_repr:
        h.update(b"\x00")
        h.update(schema_repr.encode("utf-8"))
    return h.hexdigest()[:16]


def cached_call(
    model: str,
    system: str,
    user: str,
    cache_dir: str | Path,
    call_fn: Callable[[], str],
    *,
    schema_repr: str = "",
) -> str:
    """Return cached LLM response if present, else invoke `call_fn` and cache it.

    Args:
        model:       LiteLLM model id (part of the cache key).
        system:      system prompt (part of the cache key).
        user:        user message (part of the cache key).
        cache_dir:   directory to read/write cache files.
        call_fn:     zero-arg callable returning the response string;
                     invoked only on cache miss.
        schema_repr: optional canonical JSON-Schema string for the Pydantic
                     model used in this call. Included in the cache key so
                     schema-mode and free-form responses to the same prompt
                     don't collide.
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    k = _key(model, system, user, schema_repr)
    cf = cdir / f"{k}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))["response"]
        except (json.JSONDecodeError, KeyError):
            pass
    response = call_fn()
    cf.write_text(
        json.dumps({"model": model, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )
    return response
