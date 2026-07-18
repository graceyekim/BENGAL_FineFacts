"""`{{ var }}` variable substitution in prompts. Unknown variables pass
through as literal text."""

from __future__ import annotations

import re


_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def has_template(prompt: str) -> bool:
    """True iff `prompt` contains `{{ var }}` markers we will substitute."""
    return _TEMPLATE_RE.search(prompt) is not None


def render(prompt: str, context: dict) -> str:
    """Replace every `{{ var }}` in `prompt` with `str(context[var])`.

    Unknown variables are left as the literal `{{ var }}` text so the user
    can spot them in the LLM response. Returns the rendered string; original
    prompt is unchanged if no markers match.
    """
    def sub(match):
        var = match.group(1)
        if var in context:
            return str(context[var])
        return match.group(0)
    return _TEMPLATE_RE.sub(sub, prompt)
