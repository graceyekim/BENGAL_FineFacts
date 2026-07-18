"""LLM call wrapper — LiteLLM multi-provider with exponential-backoff retry.

`schema` accepts a Pydantic BaseModel subclass and is forwarded to LiteLLM as
`response_format=`. LiteLLM translates this to each provider's native
structured-output mechanism — OpenAI's `response_format`, Anthropic's
`tool_use`, etc. Providers without strict structured output may fall back to
JSON-object mode; local validation in `extractor` catches any drift.

Retry strategy: full-jitter exponential backoff. Each retry waits
`random.uniform(0, base * 2**attempt)` seconds, capped at 60s. This avoids
the synchronized-thundering-herd that fixed `2**attempt` produces under
n_workers > 1 — when one worker hits a rate limit, multiple workers tend to
retry at exactly the same time and re-trigger the limit.
"""

from __future__ import annotations

import random
import time


_BASE_BACKOFF = 1.0
_MAX_BACKOFF = 60.0


def _sleep_with_jitter(attempt: int) -> None:
    """Full-jitter exponential backoff: uniform[0, base * 2**attempt], capped."""
    ceiling = min(_MAX_BACKOFF, _BASE_BACKOFF * (2 ** attempt))
    time.sleep(random.uniform(0, ceiling))


def call_llm(model, system, user, max_retries=5, schema=None, tracker=None):
    """Single LLM call via LiteLLM, retrying on transient failures.

    Args:
        schema:  optional Pydantic BaseModel subclass. When set, the response is
                 routed through the provider's structured-output API.
        tracker: optional CostTracker; when provided, the per-call cost is
                 added after each successful call. Models without a price
                 entry in LiteLLM's registry are recorded as $0 (best-effort).
    """
    import litellm
    last: Exception | None = None
    kwargs = {
        "model": model,
        "max_tokens": 8192,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if schema is not None:
        kwargs["response_format"] = schema
    for i in range(max_retries):
        try:
            r = litellm.completion(**kwargs)
            if tracker is not None:
                try:
                    cost = litellm.completion_cost(completion_response=r)
                    tracker.add(cost or 0.0)
                except Exception:
                    tracker.add(0.0)  # best-effort: skip if cost unavailable
            return r.choices[0].message.content
        except Exception as e:
            last = e
            if i < max_retries - 1:
                _sleep_with_jitter(i)
    raise RuntimeError(f"LLM failed: {last}")
