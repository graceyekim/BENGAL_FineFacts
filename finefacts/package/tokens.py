"""Token counting and truncation to a model's context window.

Falls back to `litellm.token_counter` when available, then to `len(text)//4`.
Context window comes from `litellm.model_cost`, then `_KNOWN_CONTEXT_WINDOWS`,
then `_DEFAULT_CONTEXT_WINDOW = 8000`. Conservative on ambiguity.
"""

from __future__ import annotations


_DEFAULT_CONTEXT_WINDOW = 8000

_KNOWN_CONTEXT_WINDOWS = {
    # Anthropic Claude — all 4-family + 3.5 are 200K
    "claude-3-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-fable": 200_000,
    # OpenAI GPT-4o family
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_000_000,
    "o1": 128_000,
    "o3": 200_000,
    # Google Gemini
    "gemini-1.5": 2_000_000,
    "gemini-2": 1_000_000,
    # Open-weight defaults
    "qwen": 32_000,
    "llama-3": 8_000,
    "mistral": 32_000,
}


def model_context_window(model: str) -> int:
    """Best-effort lookup of the model's max input-token budget."""
    if not model:
        return _DEFAULT_CONTEXT_WINDOW

    # 1. Ask LiteLLM if it knows
    try:
        import litellm
        # LiteLLM accepts the full id ("anthropic/claude-...") or the short form.
        for key in (model, model.split("/")[-1]):
            info = litellm.model_cost.get(key)
            if info:
                v = info.get("max_input_tokens") or info.get("max_tokens")
                if v:
                    return int(v)
    except Exception:
        pass

    # 2. Substring match against the known-windows table
    ml = model.lower()
    for stem, n in _KNOWN_CONTEXT_WINDOWS.items():
        if stem in ml:
            return n

    return _DEFAULT_CONTEXT_WINDOW


def count_tokens(text: str, model: str) -> int:
    """Best-effort token count. Falls back to chars / 4 when LiteLLM is unavailable."""
    if not text:
        return 0
    try:
        import litellm
        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


def truncate_to_budget(text: str, model: str, budget: int) -> str:
    """Return the longest prefix of `text` whose token count is ≤ `budget`.

    Uses a binary search over character positions for log-N token counts.
    """
    if budget <= 0 or not text:
        return ""
    if count_tokens(text, model) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid], model) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def article_budget(model: str, system_prompt: str, *,
                   max_input_tokens: int | None = None,
                   reserve_output: int = 2048,
                   safety_margin: int = 256) -> int:
    """How many tokens of article text we can fit.

    Args:
        model:             target model id (used for context-window lookup
                           and tokenization).
        system_prompt:     the system prompt whose token count we must reserve.
        max_input_tokens:  cap to honor if set; otherwise use the model's window.
        reserve_output:    tokens to leave for the model's response.
        safety_margin:     extra slack for chat-template overhead, role tokens,
                           etc. Lossy budget — better than overflow.

    Returns the article-text token budget (always ≥ 0).
    """
    window = max_input_tokens if max_input_tokens else model_context_window(model)
    used_by_system = count_tokens(system_prompt, model)
    return max(0, window - used_by_system - reserve_output - safety_margin)
