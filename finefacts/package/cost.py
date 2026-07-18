"""Cost estimation + interactive confirmation for finefacts.

Uses LiteLLM's `cost_per_token` so pricing stays current across providers
(claude-*, gpt-*, gemini/*, ...).
"""

from __future__ import annotations

import sys


def _toks(text: str, model: str | None = None) -> int:
    """Token count. Uses LiteLLM's provider-aware tokenizer when available,
    falls back to chars/4 for English-like text."""
    if not text:
        return 0
    if model:
        try:
            import litellm
            n = litellm.token_counter(model=model, text=text)
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    return max(1, len(text) // 4)


def _per_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Get cost in USD for (prompt_tokens, completion_tokens) on `model` via LiteLLM."""
    try:
        import litellm
        c_in, c_out = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(c_in) + float(c_out)
    except Exception:
        return 0.0


def estimate_extract(
    corpus_iter,
    prompt: str,
    *,
    model: str = "claude-sonnet-4-6",
    max_article_chars: int = 6000,
    expected_output_tokens: int = 2000,
    limit: int | None = None,
) -> dict:
    """Estimate cost of running ff.extract() on `corpus_iter`.

    Does NOT make API calls. Iterates `corpus_iter` once and sums tokens.
    """
    prompt_t = _toks(prompt, model)
    n = 0
    in_total = 0
    for art in corpus_iter:
        text = (art.get("text") or art.get("body") or art.get("content") or "")[:max_article_chars]
        if len(text) < 100:
            continue
        title = art.get("title") or art.get("headline") or ""
        user_t = _toks(f"Title: {title}\n\n{text}", model)
        in_total += prompt_t + user_t
        n += 1
        if limit and n >= limit:
            break
    out_total = n * expected_output_tokens
    cost = _per_token_cost(model, in_total, out_total)
    return {
        "model": model,
        "n_articles": n,
        "input_tokens_total": in_total,
        "output_tokens_total": out_total,
        "cost_usd": cost,
        "per_article_usd": (cost / n) if n else 0.0,
    }


def estimate_optimize(
    sample_corpus,
    initial_prompt: str,
    *,
    iterations: int = 7,
    sample_size: int = 25,
    judge_model: str = "claude-sonnet-4-6",
    target_model: str = "claude-sonnet-4-6",
    rubric_chars: int = 4000,
    expected_extract_output_tokens: int = 2000,
    expected_judge_output_tokens: int = 1500,
    expected_revise_output_tokens: int = 3000,
) -> dict:
    """Estimate cost of running ff.optimize_prompt()."""
    sample = list(sample_corpus)[:sample_size]
    extract_est = estimate_extract(
        sample, initial_prompt, model=target_model,
        expected_output_tokens=expected_extract_output_tokens,
    )
    extract_per_iter = extract_est["cost_usd"]

    judge_in_per_article = _toks(initial_prompt) + rubric_chars // 4 + expected_extract_output_tokens
    judge_in_total = judge_in_per_article * extract_est["n_articles"]
    judge_out_total = expected_judge_output_tokens * extract_est["n_articles"]
    judge_per_iter = _per_token_cost(judge_model, judge_in_total, judge_out_total)

    revise_in = _toks(initial_prompt) + 4000
    revise_per_iter = _per_token_cost(
        judge_model, revise_in, expected_revise_output_tokens,
    )

    per_iter = extract_per_iter + judge_per_iter + revise_per_iter
    total = per_iter * iterations
    return {
        "judge_model": judge_model,
        "target_model": target_model,
        "iterations": iterations,
        "sample_size": extract_est["n_articles"],
        "per_iteration_usd": per_iter,
        "extract_per_iter_usd": extract_per_iter,
        "judge_per_iter_usd": judge_per_iter,
        "revise_per_iter_usd": revise_per_iter,
        "total_usd": total,
    }


def estimate_distill(
    corpus_iter,
    prompt: str,
    *,
    gold_size: int = 1000,
    model: str = "claude-sonnet-4-6",
    max_article_chars: int = 6000,
) -> dict:
    """Estimate cost of `ff.extract(..., distill=True)`.

    Only the paid-API gold-generation stage costs money. Local stages (train,
    merge, distilled inference) are GPU time, not API cost — reported as such.
    """
    gold_est = estimate_extract(
        corpus_iter, prompt, model=model,
        max_article_chars=max_article_chars, limit=gold_size,
    )
    return {
        "gold_gen_cost_usd": gold_est["cost_usd"],
        "gold_articles": gold_est["n_articles"],
        "gpu_time_note": "Stages 3–5 (train + merge + inference) run locally on your GPU. No API cost.",
        "total_api_cost_usd": gold_est["cost_usd"],
    }


# ── pretty printing + interactive confirm ───────────────────────────


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:,.4f}"


def print_extract_estimate(est: dict, label: str = "extract") -> None:
    print(f"\n[finefacts] 📊 Cost estimate ({label}):", file=sys.stderr)
    print(f"  Model:          {est['model']}", file=sys.stderr)
    print(f"  Articles:       {est['n_articles']:,}", file=sys.stderr)
    print(f"  Input tokens:   ~{est['input_tokens_total']:,}", file=sys.stderr)
    print(f"  Output tokens:  ~{est['output_tokens_total']:,}", file=sys.stderr)
    print(f"  Estimated:      {_fmt_usd(est['cost_usd'])} "
          f"({_fmt_usd(est['per_article_usd'])} / article)", file=sys.stderr)


def print_optimize_estimate(est: dict) -> None:
    print(f"\n[finefacts] 📊 Cost estimate (optimize_prompt):", file=sys.stderr)
    print(f"  Iterations:        {est['iterations']}", file=sys.stderr)
    print(f"  Sample per iter:   {est['sample_size']} articles", file=sys.stderr)
    print(f"  Per-iter extract:  {_fmt_usd(est['extract_per_iter_usd'])}", file=sys.stderr)
    print(f"  Per-iter judge:    {_fmt_usd(est['judge_per_iter_usd'])}", file=sys.stderr)
    print(f"  Per-iter revise:   {_fmt_usd(est['revise_per_iter_usd'])}", file=sys.stderr)
    print(f"  Per iter total:    {_fmt_usd(est['per_iteration_usd'])}", file=sys.stderr)
    print(f"  GRAND TOTAL:       {_fmt_usd(est['total_usd'])}", file=sys.stderr)


def print_distill_estimate(est: dict) -> None:
    print(f"\n[finefacts] 📊 Cost estimate (distill):", file=sys.stderr)
    print(f"  Gold-gen articles: {est['gold_articles']:,}", file=sys.stderr)
    print(f"  Gold-gen API cost: {_fmt_usd(est['gold_gen_cost_usd'])}", file=sys.stderr)
    print(f"  {est['gpu_time_note']}", file=sys.stderr)
    print(f"  TOTAL API COST:    {_fmt_usd(est['total_api_cost_usd'])}", file=sys.stderr)


def confirm_or_abort(prompt_text: str = "Proceed?") -> None:
    """Interactive y/N prompt. Non-tty → auto-proceeds (for scripts/sbatch)."""
    if not sys.stdin.isatty():
        print(f"[finefacts] (non-interactive — auto-proceeding)", file=sys.stderr)
        return
    try:
        resp = input(f"\n[finefacts] {prompt_text} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[finefacts] Aborted.", file=sys.stderr)
        sys.exit(1)
    if resp not in ("y", "yes"):
        print("[finefacts] Aborted.", file=sys.stderr)
        sys.exit(1)
