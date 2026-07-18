"""Paid-API extraction worker.

`extract_paid` handles single-shot and chained prompts (list[str]); each
stage's user message includes the article plus the prior stage's raw
output. `extract_paid_parallel` handles a dict of independent prompts
producing keyed outputs. Both accept `n_workers` for a thread-pool over
per-article LLM calls.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import llms  # imported as module so test monkey-patching propagates
from .cache import cached_call
from .corpus import iter_corpus
from .multilingual import translation_system_prompt
from .parsing import article_meta, parse_json
from .template import has_template, render
from .tokens import article_budget, truncate_to_budget


def _wrap_progress(iterator, progress, label):
    """Optionally wrap an iterator with a tqdm progress bar.

    `progress`:
        None  — auto: enabled if stderr is a TTY (interactive), off otherwise
        True  — always on
        False — always off

    Silent-fallback when tqdm is unavailable — never fails the extraction over
    a missing UI dep.
    """
    if progress is False:
        return iterator
    if progress is None:
        import sys as _sys
        if not _sys.stderr.isatty():
            return iterator
    try:
        from tqdm import tqdm
    except ImportError:
        return iterator
    return tqdm(iterator, desc=label, unit="art", dynamic_ncols=True, leave=True)


def _schema_repr(schema):
    """Canonical JSON-Schema string for cache keying. None / falsy → empty."""
    if schema is None:
        return ""
    try:
        return json.dumps(schema.model_json_schema(), sort_keys=True)
    except AttributeError:
        # Not a Pydantic BaseModel — caller should have validated already, but
        # don't blow up the cache lookup if they didn't.
        return repr(schema)


def _validate(parsed, schema):
    """Validate `parsed` against a Pydantic schema. Returns (validated_dict, err_or_None)."""
    if schema is None or parsed is None:
        return parsed, None
    try:
        instance = schema.model_validate(parsed)
        # Round-trip through model_dump so users get a clean dict (not a model instance)
        return instance.model_dump(mode="json"), None
    except Exception as e:
        return parsed, f"schema_validation_error: {type(e).__name__}: {e}"


def _call_one(model, system, user, cache_dir, schema=None, tracker=None, engine=None):
    """Single LLM call, optionally cached. `schema` is forwarded to LiteLLM as
    `response_format` and included in the cache key. `tracker`, if set,
    accumulates per-call costs only on cache misses (real API calls).
    `engine`, if set, routes the call through a custom engine (local
    Transformers / vLLM / llama.cpp / user-supplied) instead of LiteLLM."""
    schema_repr = _schema_repr(schema)

    def _generate():
        if engine is not None:
            return engine.generate(model, system, user, schema=schema, tracker=tracker)
        return llms.call_llm(model, system, user, schema=schema, tracker=tracker)

    if cache_dir:
        return cached_call(model, system, user, cache_dir, _generate,
                           schema_repr=schema_repr)
    return _generate()


def _maybe_pre_translate(text, model, target_lang, cache_dir, tracker=None, engine=None):
    """If `target_lang` is set, run a translation LLM call and return
    (translated_text, translation). Otherwise return (text, None).

    Cached via the persistent LLM-call cache — same article + same target
    language reuses the prior translation without an extra API call.
    """
    if not target_lang:
        return text, None
    sys_prompt = translation_system_prompt(target_lang)
    translated = _call_one(model, sys_prompt, text, cache_dir, schema=None,
                           tracker=tracker, engine=engine)
    return translated, translated


# ── per-article work functions ──────────────────────────────────────


def _truncate_article(art, model, prompts_for_budget, max_article_chars,
                      max_input_tokens):
    """Pull article text, apply char ceiling, then truncate to model token budget.

    `prompts_for_budget` is a list of system prompts; we budget against the
    LONGEST one so the chosen text fits all stages.
    """
    raw = art.get("text") or art.get("body") or art.get("content") or ""
    if max_article_chars and len(raw) > max_article_chars:
        raw = raw[:max_article_chars]
    if not raw:
        return raw
    longest_system = max(prompts_for_budget, key=len, default="")
    budget = article_budget(
        model, longest_system, max_input_tokens=max_input_tokens,
    )
    return truncate_to_budget(raw, model, budget)


def _process_chained(art, prompts, model, max_article_chars, cache_dir, out_dir,
                     schema=None, max_input_tokens=None,
                     translate_to=None, tracker=None, engine=None):
    """Process one article through a CHAINED prompt list.

    `schema` applies to the FINAL stage only. Intermediate stages return
    free-form text passed forward to the next stage.

    `translate_to`, if set, runs a pre-translation LLM call and feeds the
    translated text into the user's chain. The translation is preserved as a
    `translation` field in the output record.

    Returns (path: Path, rec: dict) for caller to write, or None to skip.
    """
    text = _truncate_article(art, model, prompts, max_article_chars, max_input_tokens)
    if len(text) < 100:
        return None
    m = article_meta(art, text)
    path = out_dir / f"{m['article_id']}.json"
    if path.exists():
        return None

    # Optional pre-translation stage (translate-first cross-lingual mode).
    text, translation = _maybe_pre_translate(text, model, translate_to, cache_dir,
                                              tracker, engine)

    user_base = f"Title: {m['title']}\n\n{text}"
    stages = []
    final_raw = ""
    err = None
    try:
        current_user = user_base
        last_idx = len(prompts) - 1
        for i, p in enumerate(prompts):
            stage_schema = schema if i == last_idx else None
            rendered = render(p, art) if has_template(p) else p
            raw = _call_one(model, rendered, current_user, cache_dir,
                            schema=stage_schema, tracker=tracker, engine=engine)
            stages.append(raw)
            current_user = f"{user_base}\n\n--- Prior stage output ---\n{raw}"
        final_raw = stages[-1]
        ext = parse_json(final_raw)
        if ext is None:
            err = "json_parse_error"
        else:
            ext, validate_err = _validate(ext, schema)
            err = validate_err
    except Exception as e:
        ext = None
        err = f"{type(e).__name__}: {e}"

    rec = {**m, "extraction_date": date.today().isoformat(),
           "model": model, "extracted": ext}
    # Hoist self-reported confidence out of the extracted JSON into a
    # top-level field so callers can filter/sort on it directly.
    if isinstance(ext, dict) and "confidence" in ext:
        try:
            rec["confidence"] = float(ext["confidence"])
        except (TypeError, ValueError):
            pass
    if translation is not None:
        rec["translated_to"] = translate_to
        rec["translation"] = translation[:50000]
    if len(prompts) > 1:
        rec["_chain_stages"] = [
            {"prompt_idx": i, "raw": s[:5000]} for i, s in enumerate(stages)
        ]
    if err:
        rec["error"] = err
        rec["raw_response"] = (final_raw or "")[:5000]
    return path, rec


def _process_parallel(art, prompts_dict, model, max_article_chars, cache_dir, out_dir,
                      schema=None, max_input_tokens=None,
                      translate_to=None, tracker=None, engine=None):
    """Process one article through N PARALLEL/INDEPENDENT prompts.

    `schema` accepts three shapes:
      - None                              — no schema enforcement
      - a single Pydantic BaseModel       — same schema applied to every key
      - dict[str, BaseModel]              — per-key schemas; keys must match
                                            prompts_dict (or be a strict subset;
                                            unkeyed prompts get no schema)

    Returns (path: Path, rec: dict) for caller to write, or None to skip.
    """
    text = _truncate_article(
        art, model, list(prompts_dict.values()), max_article_chars, max_input_tokens,
    )
    if len(text) < 100:
        return None
    m = article_meta(art, text)
    path = out_dir / f"{m['article_id']}.json"
    if path.exists():
        return None

    # Optional pre-translation stage — done ONCE per article, then all
    # parallel prompts see the translated text.
    text, translation = _maybe_pre_translate(text, model, translate_to, cache_dir,
                                              tracker, engine)

    # Resolve per-key schemas once for this article.
    def schema_for(key):
        if isinstance(schema, dict):
            return schema.get(key)
        return schema

    user = f"Title: {m['title']}\n\n{text}"
    extracted = {}
    errors = {}
    raws = {}
    for key, prompt in prompts_dict.items():
        key_schema = schema_for(key)
        try:
            rendered = render(prompt, art) if has_template(prompt) else prompt
            raw = _call_one(model, rendered, user, cache_dir,
                            schema=key_schema, tracker=tracker, engine=engine)
            raws[key] = raw
            parsed = parse_json(raw)
            if parsed is None:
                errors[key] = "json_parse_error"
                extracted[key] = None
            else:
                validated, validate_err = _validate(parsed, key_schema)
                extracted[key] = validated
                if validate_err:
                    errors[key] = validate_err
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"
            extracted[key] = None
            raws[key] = ""

    rec = {**m, "extraction_date": date.today().isoformat(),
           "model": model, "extracted": extracted,
           "_parallel_keys": list(prompts_dict.keys())}
    # Hoist self-reported confidence: min across keys, so the scalar shape
    # matches chained mode. Per-key values remain inside `extracted[key]`.
    confs = []
    for v in extracted.values():
        if isinstance(v, dict) and "confidence" in v:
            try:
                confs.append(float(v["confidence"]))
            except (TypeError, ValueError):
                pass
    if confs:
        rec["confidence"] = min(confs)
    if translation is not None:
        rec["translated_to"] = translate_to
        rec["translation"] = translation[:50000]
    if errors:
        rec["_parallel_errors"] = errors
        rec["_parallel_raws"] = {k: r[:5000] for k, r in raws.items() if k in errors}
    return path, rec


# ── bounded-concurrency runner ──────────────────────────────────────


def _run(arts_iter: Iterable, work_fn: Callable, n_workers: int,
         limit: int | None, tracker=None) -> Iterator:
    """Yield non-None results of `work_fn(art)` over `arts_iter`.

    Honors n_workers (>= 1) by keeping that many work units in flight. When
    `limit` is set, stops yielding once `limit` results have been produced
    (in-flight tasks may complete but their results are dropped — they remain
    in the LLM cache, so a rerun will reuse them).

    When `tracker` is set and reports `exhausted()`, stops submitting new
    work. In-flight tasks complete normally and their results are still
    yielded — only NEW submissions are blocked. This bounds the cost
    overshoot to (n_workers − 1) calls worth.
    """
    if n_workers <= 1:
        n_done = 0
        for art in arts_iter:
            if tracker is not None and tracker.exhausted():
                return
            r = work_fn(art)
            if r is None:
                continue
            yield r
            n_done += 1
            if limit and n_done >= limit:
                return
        return

    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        in_flight = set()
        try:
            for art in arts_iter:
                if limit and n_done >= limit:
                    break
                if tracker is not None and tracker.exhausted():
                    break
                while len(in_flight) >= n_workers:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        r = fut.result()
                        if r is not None:
                            yield r
                            n_done += 1
                            if limit and n_done >= limit:
                                break
                    if limit and n_done >= limit:
                        break
                if limit and n_done >= limit:
                    break
                if tracker is not None and tracker.exhausted():
                    break
                in_flight.add(ex.submit(work_fn, art))

            while in_flight and not (limit and n_done >= limit):
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    r = fut.result()
                    if r is not None:
                        yield r
                        n_done += 1
                        if limit and n_done >= limit:
                            break
        finally:
            for f in in_flight:
                f.cancel()


# ── public extractors ───────────────────────────────────────────────


_CANONICAL_FIELDS = {"text", "title", "id", "article_id", "url",
                     "source_url", "source_domain", "base_url"}


def _alias_iter(corpus, fields):
    """Wrap iter_corpus, copying user-named columns to canonical keys.

    `fields = {"text": "body", "id": "post_id"}` means: in each article,
    set art["text"] = art["body"], art["id"] = art["post_id"]. Canonical
    keys already present in the article are preserved (alias never overwrites).
    """
    for art in iter_corpus(corpus):
        if not fields:
            yield art
            continue
        a = dict(art)
        for canonical, source in fields.items():
            if canonical not in a and source in a:
                a[canonical] = a[source]
        yield a


def _filter_iter(art_iter, filter_fn, skip_counter):
    """Wrap `art_iter` with a user-provided predicate.

    `skip_counter` is a mutable list `[int]` that gets incremented per skip
    so callers can read the total after iteration exhausts. On predicate
    exception we conservatively KEEP the article (over-include > silent drop).
    """
    if filter_fn is None:
        yield from art_iter
        return
    for art in art_iter:
        try:
            keep = bool(filter_fn(art))
        except Exception:
            keep = True
        if keep:
            yield art
        else:
            skip_counter[0] += 1


def extract_paid(corpus, prompts, output, *, model, max_article_chars,
                 limit=None, cache_dir=None, n_workers: int = 1, schema=None,
                 max_input_tokens=None, translate_to=None, tracker=None,
                 fields=None, filter_fn=None, filter_skip_counter=None,
                 engine=None, progress=None):
    """Run paid LLM extraction on `corpus`, writing per-article JSON to `output`.

    Args:
        prompts:     list of system prompts.
                     len == 1 → single-shot.
                     len > 1  → chained: stage N's user message includes the article
                                + stage N-1's raw output.
        cache_dir:   if set, cache LLM responses by (model, system, user) hash —
                     gives free resume on reruns with identical inputs.
        limit:       stop after writing this many new article files. With
                     n_workers > 1, `limit` is best-effort and may overshoot by
                     up to n_workers - 1 on the API side; only `limit` files
                     are written to disk.
        n_workers:   thread-pool size for per-article LLM calls. 1 = serial
                     (default). 4–16 is reasonable for hosted-LLM calls.
    """
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    work_fn = lambda art: _process_chained(
        art, prompts, model, max_article_chars, cache_dir, out,
        schema=schema, max_input_tokens=max_input_tokens,
        translate_to=translate_to, tracker=tracker, engine=engine,
    )
    skip_counter = filter_skip_counter if filter_skip_counter is not None else [0]
    art_iter = _filter_iter(_alias_iter(corpus, fields), filter_fn, skip_counter)
    stream = _run(art_iter, work_fn, n_workers, limit, tracker)
    for path, rec in _wrap_progress(stream, progress, f"extract [{model}]"):
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_paid_parallel(corpus, prompts_dict, output, *, model, max_article_chars,
                          limit=None, cache_dir=None, n_workers: int = 1, schema=None,
                          max_input_tokens=None, translate_to=None, tracker=None,
                          fields=None, filter_fn=None, filter_skip_counter=None,
                          engine=None, progress=None):
    """Run N INDEPENDENT prompts per article; write one JSON per article.

    Each prompt in `prompts_dict` sees only the article (NOT the other prompts'
    outputs — that's chained extraction, see `extract_paid`).

    Args:
        prompts_dict: {key: system_prompt}. Output for each article will be:
                      {..., "extracted": {key: parsed_json, ...},
                            "_parallel_errors": {key: err_msg, ...}}
        cache_dir:    optional persistent cache for (model, prompt, user).
        limit:        stop after writing this many new article files.
        n_workers:    thread-pool size; see `extract_paid` for semantics.
    """
    if not isinstance(prompts_dict, dict) or not prompts_dict:
        raise ValueError("prompts_dict must be a non-empty dict of {name: prompt}")
    for k, v in prompts_dict.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"prompts_dict keys must be non-empty strings; got {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"prompts_dict[{k!r}] must be a non-empty string")

    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    work_fn = lambda art: _process_parallel(
        art, prompts_dict, model, max_article_chars, cache_dir, out,
        schema=schema, max_input_tokens=max_input_tokens,
        translate_to=translate_to, tracker=tracker, engine=engine,
    )
    skip_counter = filter_skip_counter if filter_skip_counter is not None else [0]
    art_iter = _filter_iter(_alias_iter(corpus, fields), filter_fn, skip_counter)
    stream = _run(art_iter, work_fn, n_workers, limit, tracker)
    for path, rec in _wrap_progress(stream, progress, f"extract [{model}]"):
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
