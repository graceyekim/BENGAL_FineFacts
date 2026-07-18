"""LLM engine abstraction.

`ff.extract(provider=)` selects the engine. Default (`None` or `"litellm"`)
routes hosted providers via LiteLLM. Any object with
`generate(model, system, user, *, schema=None, tracker=None) -> str`
also qualifies.
"""

from __future__ import annotations


class LiteLLMEngine:
    """Default engine — routes to any provider LiteLLM supports."""

    def generate(self, model, system, user, *, schema=None, tracker=None):
        from . import llms
        return llms.call_llm(model, system, user, schema=schema, tracker=tracker)


def resolve_engine(provider):
    """Turn the user's `provider=` kwarg into an engine instance (or None).

    - `None` or `"litellm"` → None (caller uses the default llms.call_llm path)
    - Any object with a `generate` attr → assume it's already an engine
    - Anything else → ValueError
    """
    if provider is None or provider == "litellm":
        return None
    if hasattr(provider, "generate"):
        return provider
    raise ValueError(
        f"provider must be None, \"litellm\", or an object with a "
        f"`generate(model, system, user, *, schema, tracker) -> str` "
        f"method; got {type(provider).__name__}"
    )
