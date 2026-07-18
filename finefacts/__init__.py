from .package import (
    extract,
    optimize_prompt,
    evaluate,
    compare_by_rubric,
    load_rubric,
    estimate_cost,
    confirm_or_abort,
    list_runs,
    load,
    compare_runs,
    match_by_field,
    match_by_keywords,
    match_by_regex,
    to_csv,
    to_jsonl,
    retry_failed,
    show,
    train,
    keyword_filter,
    llm_filter,
    compose_filters,
    embed,
    cluster,
    detect_syndication,
    search,
)

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("finefacts")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "extract",
    "optimize_prompt",
    "evaluate",
    "compare_by_rubric",
    "load_rubric",
    "estimate_cost",
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
    "embed",
    "cluster",
    "detect_syndication",
    "search",
]
