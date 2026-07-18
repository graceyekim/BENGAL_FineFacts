"""Rubric-based prompt optimization and evaluation.

`optimize_prompt` iterates: extract on a sample, judge against a weighted
rubric, revise the weakest criterion, repeat. `evaluate` runs the judge
step alone. `compare_by_rubric` scores two runs side-by-side. `load_rubric`
reads a rubric YAML (None returns the bundled generic rubric).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from . import llms  # imported as module so test monkey-patching propagates
from .cache import cached_call
from .corpus import iter_corpus
from .distill import index_corpus_text
from .log import configure as _configure_log, get_logger
from .parsing import article_meta, parse_json

_logger = get_logger(__name__)


# ── rubric loading ──────────────────────────────────────────────────


def load_rubric(path=None):
    """Load a rubric YAML. `path=None` returns the bundled generic rubric."""
    import yaml
    if path is None:
        text = files("finefacts.package.rubrics").joinpath("generic.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    rubric = data.get("rubric") if isinstance(data, dict) else data
    if not isinstance(rubric, list) or not rubric:
        raise ValueError(f"Rubric YAML at {path!r} must contain a non-empty `rubric` list")
    for item in rubric:
        if "name" not in item or "weight" not in item or "requirement" not in item:
            raise ValueError(
                f"Rubric entry missing required keys (name/weight/requirement): {item!r}"
            )
    return rubric


# ── judge ───────────────────────────────────────────────────────────


_JUDGE_SYSTEM = """\
You are an expert evaluator of LLM extractions. You will be given:
  1. A source article (title + text).
  2. A prompt that was used to extract structured data from the article.
  3. The extraction the LLM produced.
  4. A rubric of criteria to evaluate against.

For EACH criterion, decide whether the requirement is MET (true) or UNMET (false).
Return a JSON object mapping criterion name → {"met": true|false, "reason": "..."}.
Be strict. If you would not be confident defending the verdict in writing, mark UNMET.
Output ONLY the JSON object, no surrounding prose.
"""


def _judge_one(article_text, title, prompt, extraction_raw, rubric, judge_model, cache_dir):
    rubric_block = "\n".join(
        f"- {c['name']} (weight={c['weight']}): {c['requirement'].strip()}"
        for c in rubric
    )
    user = (
        f"=== ARTICLE TITLE ===\n{title}\n\n"
        f"=== ARTICLE TEXT ===\n{article_text}\n\n"
        f"=== PROMPT USED ===\n{prompt}\n\n"
        f"=== EXTRACTION ===\n{extraction_raw}\n\n"
        f"=== RUBRIC ===\n{rubric_block}\n\n"
        f"Return the verdict JSON now."
    )
    if cache_dir:
        raw = cached_call(
            judge_model, _JUDGE_SYSTEM, user, cache_dir,
            lambda: llms.call_llm(judge_model, _JUDGE_SYSTEM, user),
        )
    else:
        raw = llms.call_llm(judge_model, _JUDGE_SYSTEM, user)
    parsed = parse_json(raw)
    if not isinstance(parsed, dict):
        return {c["name"]: {"met": False, "reason": "judge_parse_error"} for c in rubric}
    return {
        c["name"]: {
            "met": bool(parsed.get(c["name"], {}).get("met", False)),
            "reason": str(parsed.get(c["name"], {}).get("reason", "")),
        }
        for c in rubric
    }


def _score(verdicts, rubric):
    """AutoRubric score: met-positive-weight share, minus negative-weight penalties."""
    pos_total = sum(c["weight"] for c in rubric if c["weight"] > 0)
    met_pos = sum(c["weight"] for c in rubric
                  if c["weight"] > 0 and verdicts.get(c["name"], {}).get("met", False))
    penalty = sum(-c["weight"] for c in rubric
                  if c["weight"] < 0 and verdicts.get(c["name"], {}).get("met", False))
    if pos_total <= 0:
        return 0.0
    return (met_pos - penalty) / pos_total


def _weakest_criterion(per_article_verdicts, rubric):
    """Return (criterion_dict, fail_examples) for the criterion with the most failures
    (positive-weight) or most fires (negative-weight)."""
    fail_counts = {c["name"]: 0 for c in rubric}
    fail_examples = {c["name"]: [] for c in rubric}
    for art_id, verdicts in per_article_verdicts.items():
        for c in rubric:
            v = verdicts.get(c["name"], {})
            fired = (c["weight"] > 0 and not v.get("met", False)) or \
                    (c["weight"] < 0 and v.get("met", False))
            if fired:
                fail_counts[c["name"]] += 1
                if len(fail_examples[c["name"]]) < 3:
                    fail_examples[c["name"]].append(
                        {"article_id": art_id, "reason": v.get("reason", "")[:300]}
                    )
    weighted = {c["name"]: fail_counts[c["name"]] * abs(c["weight"]) for c in rubric}
    if not weighted or max(weighted.values()) == 0:
        return None, {}
    worst_name = max(weighted, key=weighted.get)
    worst = next(c for c in rubric if c["name"] == worst_name)
    return worst, fail_examples[worst_name]


# ── improver ────────────────────────────────────────────────────────


_IMPROVER_SYSTEM = """\
You are an expert at writing system prompts for LLM extraction tasks. You will
receive a current prompt, a criterion that the prompt's outputs are failing on,
and concrete failure examples. Return ONLY the revised prompt as plain text —
no commentary, no markdown fences, no explanation. Keep the prompt's overall
intent and schema unchanged; add or sharpen instructions ONLY to fix the
specified weakness.
"""


def _revise_prompt(current_prompt, criterion, fail_examples, improver_model, cache_dir):
    ex_block = "\n\n".join(
        f"Example {i + 1} ({e['article_id']}): {e['reason']}"
        for i, e in enumerate(fail_examples)
    )
    user = (
        f"=== CURRENT PROMPT ===\n{current_prompt}\n\n"
        f"=== WEAK CRITERION ===\n"
        f"Name: {criterion['name']}\n"
        f"Weight: {criterion['weight']}\n"
        f"Requirement: {criterion['requirement'].strip()}\n\n"
        f"=== FAILURE EXAMPLES (judge's reasoning) ===\n{ex_block}\n\n"
        f"Return the revised prompt now."
    )
    revised = cached_call(
        improver_model, _IMPROVER_SYSTEM, user, cache_dir,
        lambda: llms.call_llm(improver_model, _IMPROVER_SYSTEM, user),
    )
    return revised.strip()


# ── main loop ───────────────────────────────────────────────────────


def optimize_prompt(*,
                    initial_prompt: str,
                    sample_corpus,
                    iterations: int = 7,
                    judge_model: str = "claude-sonnet-4-6",
                    target_model: str = "claude-sonnet-4-6",
                    improver_model: str | None = None,
                    rubric=None,
                    work_dir="./optimize_work",
                    max_article_chars: int = 6000,
                    sample_size: int = 25):
    """Iteratively improve an extraction prompt against a weighted rubric.

    Args:
        initial_prompt:  starting system prompt to extract with.
        sample_corpus:   iterable of dicts or glob path; first `sample_size`
                         articles are used per iteration.
        iterations:      number of revise-and-rescore rounds.
        judge_model:     LLM that grades each extraction against the rubric.
        target_model:    LLM whose extractions are being optimized.
        improver_model:  LLM that revises the prompt (defaults to `judge_model`).
        rubric:          path to a rubric YAML, OR an already-loaded rubric
                         (list of {name, weight, requirement}), OR None for the
                         bundled generic rubric.
        work_dir:        scratch dir for caches, per-iteration prompts/verdicts.
        max_article_chars: truncate articles before sending.
        sample_size:     how many articles per iteration.

    Returns:
        {
          "best_prompt":     str,
          "best_iteration":  int,
          "best_score":      float,
          "history":         [{iteration, prompt, score, weakest_criterion}, ...],
          "work_dir":        str,
        }
    """
    if not isinstance(initial_prompt, str) or not initial_prompt.strip():
        raise ValueError("initial_prompt must be a non-empty string")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    improver_model = improver_model or judge_model
    rubric_list = rubric if isinstance(rubric, list) else load_rubric(rubric)

    work = Path(work_dir).resolve()
    (work / "prompts").mkdir(parents=True, exist_ok=True)
    (work / "verdicts").mkdir(parents=True, exist_ok=True)
    cache_dir = work / ".cache"

    sample = []
    for art in iter_corpus(sample_corpus):
        text = (art.get("text") or art.get("body") or art.get("content") or "")[:max_article_chars]
        if len(text) < 100:
            continue
        sample.append({**art, "_text": text, "_meta": article_meta(art, text)})
        if len(sample) >= sample_size:
            break
    if not sample:
        raise ValueError("sample_corpus produced 0 usable articles")
    _logger.info("optimize: sample=%d articles, rubric=%d criteria",
                 len(sample), len(rubric_list))

    current_prompt = initial_prompt.strip()
    history = []
    best = {"prompt": current_prompt, "score": -float("inf"), "iter": -1}

    for it in range(iterations):
        _logger.info("optimize iter %d/%d", it + 1, iterations)
        (work / "prompts" / f"iter_{it:02d}.txt").write_text(current_prompt, encoding="utf-8")

        per_article_verdicts = {}
        per_article_scores = []
        for art in sample:
            user = f"Title: {art['_meta']['title']}\n\n{art['_text']}"
            extraction_raw = cached_call(
                target_model, current_prompt, user, cache_dir,
                lambda: llms.call_llm(target_model, current_prompt, user),
            )
            verdicts = _judge_one(
                art["_text"], art["_meta"]["title"],
                current_prompt, extraction_raw,
                rubric_list, judge_model, cache_dir,
            )
            per_article_verdicts[art["_meta"]["article_id"]] = verdicts
            per_article_scores.append(_score(verdicts, rubric_list))

        mean_score = sum(per_article_scores) / len(per_article_scores)
        worst, fail_examples = _weakest_criterion(per_article_verdicts, rubric_list)

        (work / "verdicts" / f"iter_{it:02d}.json").write_text(
            json.dumps({
                "iteration": it,
                "mean_score": mean_score,
                "weakest_criterion": worst["name"] if worst else None,
                "per_article_verdicts": per_article_verdicts,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _logger.info("  score=%.3f  weakest=%s",
                     mean_score, worst["name"] if worst else "none")

        history.append({
            "iteration": it,
            "prompt": current_prompt,
            "score": mean_score,
            "weakest_criterion": worst["name"] if worst else None,
        })
        if mean_score > best["score"]:
            best = {"prompt": current_prompt, "score": mean_score, "iter": it}

        if worst is None or it == iterations - 1:
            break  # nothing left to fix, or final iter — no need to revise
        current_prompt = _revise_prompt(
            current_prompt, worst, fail_examples, improver_model, cache_dir,
        )

    summary = {
        "best_prompt": best["prompt"],
        "best_iteration": best["iter"],
        "best_score": best["score"],
        "history": history,
        "work_dir": str(work),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (work / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (work / "best_prompt.txt").write_text(best["prompt"], encoding="utf-8")
    _logger.info("optimize done. best iter=%d score=%.3f", best["iter"], best["score"])
    return summary


# ── evaluate — grade existing extractions against a rubric ─────────────


def evaluate(
    output,
    corpus,
    *,
    prompt: str | None = None,
    rubric=None,
    judge_model: str = "claude-sonnet-4-6",
    sample_size: int | None = None,
    cache_dir=None,
    max_article_chars: int = 6000,
    verbose: bool = False,
    quiet: bool = False,
) -> dict:
    """Score existing extractions against a rubric using an LLM-as-judge.

    Same judge as `optimize_prompt` uses internally, but as a one-shot call:
    no iteration, no prompt revision — just per-record scores + a per-
    criterion pass-rate breakdown so you can compare runs side-by-side.

    Args:
        output:            path to a finefacts run dir (from `ff.extract`) OR
                           a list of already-loaded records.
        corpus:            the source corpus used to extract. Required — the
                           per-article records don't store the article body,
                           so we look up text by `article_id`.
        prompt:            the system prompt used when extracting. Given to the
                           judge for context. If None, reads `prompt_preview`
                           from the run manifest, or falls back to a stub.
        rubric:            YAML path, in-memory list, or None (→ bundled generic).
        judge_model:       LLM that scores each extraction.
        sample_size:       optionally evaluate a random sample of records; None
                           evaluates all.
        cache_dir:         optional persistent cache for judge calls. Defaults
                           to `<output>/.evaluate_cache` when `output` is a dir.
        max_article_chars: truncate article text before sending to the judge.
        verbose / quiet:   log level.

    Returns:
        {
            "mean_score":     float,
            "n_evaluated":    int,
            "per_record":     [{"article_id", "score", "verdicts"}, ...],
            "per_criterion":  {criterion_name: {"weight", "pass_rate",
                                                "n_met", "n_unmet"}, ...},
            "weakest":        criterion_name | None,
        }

    Example:
        # Score one run
        result = ff.evaluate(output="./h1/", corpus="./articles.jsonl")
        print(f"Mean score: {result['mean_score']:.3f}")

        # Compare 10 hypothesis runs
        scores = {f"h{i+1}": ff.evaluate(f"./h{i+1}/", "./articles.jsonl")
                  for i in range(10)}
        best = max(scores, key=lambda k: scores[k]["mean_score"])
    """
    from .manifest import load as _load

    rubric_list = rubric if isinstance(rubric, list) else load_rubric(rubric)

    # Resolve records + default cache dir + optional prompt from manifest
    if isinstance(output, (str, Path)):
        out_path = Path(output)
        records = _load(out_path, include_errors=False)
        if cache_dir is None:
            cache_dir = out_path / ".evaluate_cache"
        if prompt is None:
            mp = out_path / "manifest.json"
            if mp.exists():
                try:
                    prompt = json.loads(mp.read_text(encoding="utf-8")).get("prompt_preview")
                except json.JSONDecodeError:
                    pass
    else:
        records = list(output)

    if prompt is None:
        prompt = "Extract structured data from the article."

    if not records:
        raise ValueError("No records to evaluate (output dir empty or all errored).")

    if sample_size is not None and sample_size < len(records):
        import random
        records = random.sample(records, sample_size)

    _configure_log(run_name="evaluate", verbose=verbose, quiet=quiet)
    _logger.info("evaluate: %d records against %d criteria (judge=%s)",
                 len(records), len(rubric_list), judge_model)

    corpus_idx = index_corpus_text(corpus)

    per_record = []
    per_crit_counts = {c["name"]: {"met": 0, "unmet": 0} for c in rubric_list}

    for rec in records:
        aid = rec.get("article_id")
        text = corpus_idx.get(aid, "")[:max_article_chars]
        if not text:
            continue
        title = rec.get("title", "")
        extraction_json = json.dumps(rec.get("extracted"), ensure_ascii=False)

        verdicts = _judge_one(
            text, title, prompt, extraction_json,
            rubric_list, judge_model, cache_dir,
        )
        score = _score(verdicts, rubric_list)
        per_record.append({
            "article_id": aid,
            "score": score,
            "verdicts": verdicts,
        })

        for c in rubric_list:
            v = verdicts.get(c["name"], {})
            per_crit_counts[c["name"]]["met" if v.get("met") else "unmet"] += 1

    if not per_record:
        raise RuntimeError(
            "0 records were evaluated — the corpus lookup failed for every "
            "article. Check that `corpus` matches what was used at extraction "
            "time and that article_ids line up."
        )

    n = len(per_record)
    mean_score = sum(r["score"] for r in per_record) / n

    per_criterion = {}
    for c in rubric_list:
        counts = per_crit_counts[c["name"]]
        total = counts["met"] + counts["unmet"]
        per_criterion[c["name"]] = {
            "weight": c["weight"],
            "pass_rate": counts["met"] / total if total else 0.0,
            "n_met": counts["met"],
            "n_unmet": counts["unmet"],
        }

    weakest_verdicts = {r["article_id"]: r["verdicts"] for r in per_record}
    worst, _ = _weakest_criterion(weakest_verdicts, rubric_list)

    _logger.info("evaluate done. mean=%.3f  weakest=%s",
                 mean_score, worst["name"] if worst else "none")

    return {
        "mean_score": mean_score,
        "n_evaluated": n,
        "per_record": per_record,
        "per_criterion": per_criterion,
        "weakest": worst["name"] if worst else None,
    }


# ── compare_by_rubric — score two runs against the same rubric ────────


def compare_by_rubric(
    dir_a,
    dir_b,
    corpus,
    *,
    rubric=None,
    prompt: str | None = None,
    judge_model: str = "claude-sonnet-4-6",
    sample_size: int | None = None,
    cache_dir=None,
    max_article_chars: int = 6000,
    tolerance: float = 0.05,
    verbose: bool = False,
    quiet: bool = False,
) -> dict:
    """Score two runs against the same rubric and report side-by-side.

    Runs `evaluate()` on both directories, then reports per-criterion
    pass-rate deltas, which run wins on each criterion, and a
    one-line recommendation.

    Args:
        dir_a, dir_b:      the two run directories to compare.
        corpus:            source corpus (needed to look up article text).
        rubric:            None → bundled generic rubric. Or path/list.
        prompt:            optional system prompt to give the judge for
                           context; falls back to per-run manifest prompt_preview.
        judge_model:       LLM that scores each extraction.
        sample_size:       optional random subsample of articles for
                           speed/cost. Applied identically to both runs.
        cache_dir:         judge cache; None → `<dir_a>/.compare_by_rubric_cache`.
        max_article_chars: truncate article text before sending to judge.
        tolerance:         pass-rate delta below this counts as a tie.
                           Default 0.05 (5 percentage points).

    Returns:
        {
            "a":            evaluate(dir_a) result,
            "b":            evaluate(dir_b) result,
            "delta": {
                "mean_score":    float,          # b - a
                "per_criterion": {name: delta},
            },
            "a_wins_by_criterion": [name, ...],
            "b_wins_by_criterion": [name, ...],
            "ties":              [name, ...],
            "recommendation":    str,
        }

    Example:
        r = ff.compare_by_rubric(
            "./claude_ref/", "./qwen_base/",
            corpus="./articles.jsonl",
        )
        print(f"Mean score: A={r['a']['mean_score']:.2f}  "
              f"B={r['b']['mean_score']:.2f}  delta={r['delta']['mean_score']:+.2f}")
        print(r["recommendation"])
    """
    if cache_dir is None:
        try:
            p = Path(dir_a)
            if p.is_dir():
                cache_dir = p / ".compare_by_rubric_cache"
        except (TypeError, OSError):
            pass

    common_kwargs = dict(
        corpus=corpus, rubric=rubric, prompt=prompt,
        judge_model=judge_model, sample_size=sample_size,
        cache_dir=cache_dir, max_article_chars=max_article_chars,
        verbose=verbose, quiet=quiet,
    )
    result_a = evaluate(dir_a, **common_kwargs)
    result_b = evaluate(dir_b, **common_kwargs)

    # Per-criterion pass-rate deltas (B - A)
    per_crit_delta = {}
    a_wins, b_wins, ties = [], [], []
    for name in result_a["per_criterion"]:
        a_rate = result_a["per_criterion"][name]["pass_rate"]
        b_rate = result_b["per_criterion"][name].get("pass_rate", 0.0) \
            if name in result_b["per_criterion"] else 0.0
        delta = b_rate - a_rate
        per_crit_delta[name] = delta
        if delta > tolerance:
            b_wins.append(name)
        elif delta < -tolerance:
            a_wins.append(name)
        else:
            ties.append(name)

    mean_delta = result_b["mean_score"] - result_a["mean_score"]

    if mean_delta > tolerance:
        rec = (
            f"B scores higher ({result_b['mean_score']:.3f} vs "
            f"{result_a['mean_score']:.3f}, +{mean_delta:.3f}). "
            f"Prefer B."
        )
    elif mean_delta < -tolerance:
        rec = (
            f"A scores higher ({result_a['mean_score']:.3f} vs "
            f"{result_b['mean_score']:.3f}, {mean_delta:.3f}). "
            f"Prefer A."
        )
    else:
        rec = (
            f"Tie ({mean_delta:+.3f} within tolerance {tolerance}). "
            f"Either works — pick the cheaper / faster option."
        )

    _logger.info("compare_by_rubric: A=%.3f  B=%.3f  delta=%+.3f",
                 result_a["mean_score"], result_b["mean_score"], mean_delta)

    return {
        "a": result_a,
        "b": result_b,
        "delta": {
            "mean_score": mean_delta,
            "per_criterion": per_crit_delta,
        },
        "a_wins_by_criterion": a_wins,
        "b_wins_by_criterion": b_wins,
        "ties": ties,
        "recommendation": rec,
    }
