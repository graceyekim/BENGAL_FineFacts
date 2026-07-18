"""Public API surface — kept separate from the upstream pipeline modules
(`finefacts/config.py`, `finefacts/schemas.py`, `finefacts/utils.py`).

Entry points are re-exported at the top level:

    import finefacts as ff
    ff.extract(corpus=..., prompt=..., output=...)                    # paid
    ff.extract(corpus=..., prompt=[p1, p2], output=...)               # chained
    ff.extract(corpus=..., prompt={"facts": p1, "tags": p2}, output=...)  # parallel
    ff.estimate_cost(corpus=..., prompt=...)                          # dry-run preview
    ff.optimize_prompt(initial_prompt=..., sample_corpus=...)         # rubric loop
"""

from .api import extract
from .filter import compose_filters, keyword_filter, llm_filter
from .cost import (
    estimate_extract,
    estimate_optimize,
    estimate_distill,
    print_extract_estimate,
    print_optimize_estimate,
    print_distill_estimate,
    confirm_or_abort,
)
from .manifest import (
    compare_runs, list_runs, load,
    match_by_field, match_by_keywords, match_by_regex,
    retry_failed, show, to_csv, to_jsonl,
)
from .finetune import train
from .optimize import compare_by_rubric, evaluate, optimize_prompt, load_rubric
# Analysis subpackage — function defs only; heavy deps are lazy-imported
# inside each function. Importing finefacts without [analysis] installed is fine.
from .analysis import embed, cluster, detect_syndication, search


def estimate_cost(corpus, prompt, *, mode="extract", **kw):
    """Estimate cost of an `extract` or `optimize` run without making API calls.

    Args:
        corpus:  iterable of dicts or glob path.
        prompt:  string (single), list[str] (chained), or dict[str, str] (parallel).
        mode:    "extract" | "distill" | "optimize".
        **kw:    forwarded to the underlying estimator (model, gold_size, ...).

    Returns the estimator's dict (cost_usd, n_articles, ...).
    """
    if mode == "extract":
        # Multi-prompt (chained OR parallel) scales the cost by n_prompts; the
        # estimator only needs one representative prompt for the input-token count.
        if isinstance(prompt, list):
            sample_p, multiplier = prompt[0] if prompt else "", len(prompt)
        elif isinstance(prompt, dict):
            sample_p = next(iter(prompt.values())) if prompt else ""
            multiplier = len(prompt)
        else:
            sample_p, multiplier = prompt, 1
        est = estimate_extract(corpus, sample_p, **kw)
        if multiplier > 1:
            est["cost_usd"] *= multiplier
            est["per_article_usd"] *= multiplier
            est["output_tokens_total"] *= multiplier
        return est
    if mode == "distill":
        return estimate_distill(corpus, prompt, **kw)
    if mode == "optimize":
        return estimate_optimize(corpus, prompt, **kw)
    raise ValueError(f"mode must be 'extract', 'distill', or 'optimize'; got {mode!r}")


__all__ = [
    "extract",
    "optimize_prompt",
    "evaluate",
    "compare_by_rubric",
    "load_rubric",
    "estimate_cost",
    "estimate_extract",
    "estimate_optimize",
    "estimate_distill",
    "print_extract_estimate",
    "print_optimize_estimate",
    "print_distill_estimate",
    "confirm_or_abort",
    "list_runs",
    "load",
    "compare_runs",
    "match_by_field",
    "match_by_keywords",
    "match_by_regex",
    "to_csv",
    "to_jsonl",
    "retry_failed",
    "show",
    "train",
    "keyword_filter",
    "llm_filter",
    "compose_filters",
    # analysis (lazy-loaded heavy deps)
    "embed",
    "cluster",
    "detect_syndication",
    "search",
]
