"""`ff.extract` entry point — dispatches on prompt shape and orchestrates the
paid-API, batch, and distillation paths. All real work lives in the modules
imported below.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .corpus import iter_corpus
from .distill import (
    build_conversations,
    index_corpus_text,
    run_script,
    run_distilled,
    split_jsonl,
)
from .cost_cap import CostTracker
from .engine import resolve_engine
from .extractor import extract_paid, extract_paid_parallel
from .log import configure as _configure_log, get_logger
from .manifest import build_manifest, finalize_manifest
from .multilingual import (
    canonical_language_name,
    wrap_prompts_with_language,
)

_logger = get_logger(__name__)


def extract(corpus, prompt, output, *,
            model="claude-sonnet-4-6",
            max_article_chars=6000,
            distill=False,
            distill_model="Qwen/Qwen3-4B",
            gold_size=1000,
            work_dir=None,
            name=None,
            confirm=True,
            cache=True,
            fresh: bool = False,
            n_workers: int = 1,
            schema=None,
            max_input_tokens=None,
            output_language: str | None = None,
            translate_first: bool = False,
            max_spend: float | None = None,
            fields: dict | None = None,
            filter=None,
            provider=None,
            progress=None,
            capture_confidence: bool = False,
            batch: bool = False,
            batch_poll_interval: float = 60.0,
            batch_timeout: float | None = None,
            verbose: bool = False,
            quiet: bool = False):
    """Extract structured data from a corpus via an LLM.

    Writes one JSON file per article to `output/`, plus a manifest.

    Args:
        corpus: iterable of dicts or a glob path (csv/jsonl/jsonl.gz/parquet).
        prompt: system prompt. str = single-shot, list[str] = chained,
            dict[str, str] = parallel (each prompt sees only the article).
        output: output directory (created if missing; resumable).
        model: LiteLLM model id.
        max_article_chars: char ceiling on article text before sending.
        distill: after gold gen, run the 5-stage distillation pipeline.
        distill_model: HuggingFace hub id to fine-tune.
        gold_size: number of articles for teacher gold.
        work_dir: scratch dir for distillation artifacts.
        name: run name; defaults to basename of `output`.
        confirm: print cost estimate and prompt y/N before spending.
        cache: cache LLM responses under `output/.cache/`.
        fresh: if True, delete existing `*.json` in `output/` before running,
            so this run starts clean. `output/.cache/` is preserved unless
            `cache=False` (then it's deleted too). Default False = resume.
        n_workers: thread-pool size for per-article calls.
        schema: Pydantic model (all modes) or dict[str, model] (parallel only).
        max_input_tokens: hard cap on input tokens; None = auto from model.
        output_language: force output to this language (e.g. "en", "ko").
        translate_first: pre-translate each article to `output_language`,
            then extract on the translation.
        max_spend: USD ceiling; new submissions stop once reached.
        fields: column aliases, e.g. `{"text": "body", "id": "post_id"}`.
        filter: `Callable[[dict], bool]` — return False to skip.
        provider: `None` / `"litellm"` (default), or any object with a
            `.generate(model, system, user, *, schema, tracker) -> str` method.
        progress: tqdm bar. None = auto (TTY), True/False = force.
        capture_confidence: append a self-report confidence instruction;
            hoist the value into `record.confidence`.
        batch: submit as an Anthropic Message Batch (single-shot only, 50%
            of the standard token rate, up to 24h latency).
        batch_poll_interval: seconds between batch status polls.
        batch_timeout: max seconds to wait for the batch; None = forever.
        verbose: DEBUG-level logging.
        quiet: WARNING-only logging.

    Returns:
        The output directory as a `pathlib.Path`.
    """
    # Dispatch on `prompt` type:
    #   str         → single-shot (treated as a 1-element chain)
    #   list[str]   → chained (stage N sees stage N-1's output)
    #   dict[str,str] → parallel (each prompt independent, outputs keyed)
    if isinstance(prompt, str):
        mode = "chained"
        prompts = [prompt]
    elif isinstance(prompt, list):
        mode = "chained"
        prompts = list(prompt)
    elif isinstance(prompt, dict):
        mode = "parallel"
        prompts = dict(prompt)
    else:
        raise ValueError("prompt must be str, list[str], or dict[str, str]")

    if mode == "chained":
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("prompt list must be non-empty with non-empty strings")
    else:  # parallel
        if not prompts:
            raise ValueError("prompt dict must be non-empty")
        for k, v in prompts.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError(f"prompt dict keys must be non-empty strings; got {k!r}")
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"prompt dict[{k!r}] must be a non-empty string")

    def _is_pydantic_model(s):
        return hasattr(s, "model_json_schema") and hasattr(s, "model_validate")

    if schema is not None:
        if isinstance(schema, dict):
            # Per-key schemas — only valid in parallel mode.
            if mode != "parallel":
                raise ValueError(
                    "dict-shaped schema is only valid in parallel mode "
                    "(prompt={...}); use a single Pydantic class for chained/single."
                )
            for k, v in schema.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"schema dict keys must be strings; got {type(k).__name__}"
                    )
                if not _is_pydantic_model(v):
                    raise ValueError(
                        f"schema[{k!r}] must be a Pydantic BaseModel subclass; "
                        f"got {type(v).__name__}"
                    )
            extra = set(schema.keys()) - set(prompts.keys())
            if extra:
                raise ValueError(
                    f"schema dict has keys not present in prompt dict: {sorted(extra)}"
                )
        elif not _is_pydantic_model(schema):
            raise ValueError(
                "schema must be a Pydantic BaseModel subclass, or a "
                "dict[str, BaseModel] for per-key parallel schemas; "
                f"got {type(schema).__name__}"
            )

    if mode != "chained" and distill:
        raise NotImplementedError(
            "Parallel multi-prompt (prompt=dict) is not supported with distill=True. "
            "Use a single prompt for distillation."
        )

    # Multilingual validation.
    if translate_first and not output_language:
        raise ValueError(
            "translate_first=True requires output_language=... (which language "
            "to translate the articles into)."
        )
    if output_language is not None:
        if not isinstance(output_language, str):
            raise ValueError(
                f"output_language must be a string; got {type(output_language).__name__}"
            )
        if not output_language.strip():
            raise ValueError("output_language must be a non-empty string")
    if translate_first and distill:
        raise NotImplementedError(
            "translate_first=True is not yet supported with distill=True. "
            "Run paid extraction with translate_first first, then distill on its output."
        )

    # Batch API validation.
    if batch:
        if mode != "chained" or len(prompts) > 1:
            raise ValueError(
                "batch=True is single-shot only (prompt=str). "
                "Chained and parallel prompt modes are not yet supported."
            )
        if distill:
            raise ValueError("batch=True is incompatible with distill=True")
        if provider not in (None, "litellm"):
            raise ValueError(
                "batch=True requires the default litellm/Anthropic path "
                "(local engines can't batch)"
            )
        if translate_first:
            raise ValueError(
                "batch=True is not yet supported with translate_first. "
                "Use output_language= for a direct single-call cross-lingual extraction."
            )
    if mode == "chained" and len(prompts) > 1 and distill:
        raise NotImplementedError(
            "Chained extraction (prompt=list) is not yet supported with distill=True. "
            "Use a single prompt for distillation, or run paid chained extraction first."
        )

    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    effective_name = name or out.name
    cache_dir = (out / ".cache") if cache else None
    _configure_log(run_name=effective_name, verbose=verbose, quiet=quiet,
                   log_file=out / "run.log")

    # Direct cross-lingual: append a language directive to every system prompt
    # in place. No extra LLM call. `translate_first=True` is handled separately
    # below by passing `translate_to` to the extractor.
    if output_language is not None and not translate_first:
        prompts = wrap_prompts_with_language(prompts, output_language)

    # Confidence self-report: append an instruction; extractor hoists the
    # field out of `extracted` into a top-level `confidence` key.
    if capture_confidence:
        confidence_directive = (
            "\n\nAlso include a top-level field named \"confidence\" in your "
            "JSON output — a single float between 0.0 and 1.0 rating your "
            "confidence in the accuracy of this extraction. 1.0 = certain, "
            "0.0 = pure guess."
        )
        if isinstance(prompts, list):
            prompts = [p + confidence_directive for p in prompts]
        elif isinstance(prompts, dict):
            prompts = {k: v + confidence_directive for k, v in prompts.items()}
        else:
            prompts = prompts + confidence_directive

    if mode == "chained":
        prompt_for_manifest = "\n\n--- CHAIN STAGE ---\n\n".join(prompts)
        n_calls_per_article = len(prompts)
    else:
        prompt_for_manifest = "\n\n--- PARALLEL: {key} ---\n\n".join(
            [f"{k}\n{v}" for k, v in prompts.items()]
        )
        n_calls_per_article = len(prompts)
    if translate_first:
        n_calls_per_article += 1  # one extra translation call per article

    manifest = build_manifest(effective_name, model, prompt_for_manifest, max_article_chars,
                              distill, distill_model, gold_size)
    manifest["mode"] = mode
    manifest["n_prompts"] = n_calls_per_article
    if mode == "parallel":
        manifest["parallel_keys"] = list(prompts.keys())
    _logger.info("Starting run (mode=%s, n_prompts=%d)",
                 mode, n_calls_per_article)

    if fresh:
        removed = 0
        for p in out.glob("*.json"):
            p.unlink()
            removed += 1
        if not cache:
            cache_path = out / ".cache"
            if cache_path.exists():
                shutil.rmtree(cache_path)
        if removed:
            _logger.info("fresh=True: removed %d file(s) from %s", removed, out)

    if confirm:
        from .cost import (
            estimate_extract, estimate_distill,
            print_extract_estimate, print_distill_estimate, confirm_or_abort,
        )
        # For the estimator's input-token count, any one of the prompts is representative
        # (input is dominated by article text). Multiply by n_prompts for total cost.
        sample_prompt = prompts[0] if mode == "chained" else next(iter(prompts.values()))
        if distill:
            est = estimate_distill(iter_corpus(corpus), sample_prompt,
                                   gold_size=gold_size, model=model,
                                   max_article_chars=max_article_chars)
            print_distill_estimate(est)
        else:
            est = estimate_extract(iter_corpus(corpus), sample_prompt, model=model,
                                   max_article_chars=max_article_chars)
            if n_calls_per_article > 1:
                est["cost_usd"] *= n_calls_per_article
                est["per_article_usd"] *= n_calls_per_article
                est["output_tokens_total"] *= n_calls_per_article
            print_extract_estimate(
                est, label=f"{effective_name} ({mode} x{n_calls_per_article})",
            )
        confirm_or_abort(f"Proceed with {effective_name}?")

    # Validate `filter=` kwarg early.
    if filter is not None and not callable(filter):
        raise ValueError(
            f"filter must be a Callable[[dict], bool]; got {type(filter).__name__}"
        )

    # Validate `fields=` kwarg early.
    if fields is not None:
        if not isinstance(fields, dict):
            raise ValueError(
                f"fields must be a dict[str, str]; got {type(fields).__name__}"
            )
        for k, v in fields.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(
                    f"fields entries must be str → str; got {k!r}: {v!r}"
                )
            if not k.strip() or not v.strip():
                raise ValueError(
                    f"fields entries must be non-empty strings; got {k!r}: {v!r}"
                )

    translate_to = output_language if translate_first else None
    tracker = CostTracker(max_spend=max_spend) if max_spend is not None else None
    filter_skip_counter = [0]  # updated in-place by _filter_iter
    engine = resolve_engine(provider)

    if not distill:
        if batch:
            from .batch import run_batch
            run_batch(
                iter_corpus(corpus), prompts[0], out,
                model=model,
                max_article_chars=max_article_chars,
                max_input_tokens=max_input_tokens,
                fields=fields,
                filter_fn=filter,
                filter_skip_counter=filter_skip_counter,
                poll_interval=batch_poll_interval,
                timeout=batch_timeout,
                progress=progress,
                tracker=tracker,
            )
        elif mode == "chained":
            extract_paid(corpus, prompts, out, model=model,
                         max_article_chars=max_article_chars,
                         cache_dir=cache_dir, n_workers=n_workers, schema=schema,
                         max_input_tokens=max_input_tokens, translate_to=translate_to,
                         tracker=tracker, fields=fields,
                         filter_fn=filter, filter_skip_counter=filter_skip_counter,
                         engine=engine, progress=progress)
        else:
            extract_paid_parallel(corpus, prompts, out, model=model,
                                  max_article_chars=max_article_chars,
                                  cache_dir=cache_dir, n_workers=n_workers, schema=schema,
                                  max_input_tokens=max_input_tokens, translate_to=translate_to,
                                  tracker=tracker, fields=fields,
                                  filter_fn=filter, filter_skip_counter=filter_skip_counter,
                                  engine=engine, progress=progress)
        manifest["n_workers"] = n_workers
        manifest["max_input_tokens"] = max_input_tokens
        if batch:
            manifest["batch"] = True
        if output_language is not None:
            manifest["output_language"] = canonical_language_name(output_language)
            manifest["translate_first"] = translate_first
        if tracker is not None:
            manifest["cost_tracking"] = tracker.snapshot()
        if fields:
            manifest["fields"] = dict(fields)
        if filter is not None:
            manifest["filter"] = {
                "applied": True,
                "name": getattr(filter, "__name__", "filter"),
                "n_skipped": filter_skip_counter[0],
            }
        if engine is not None:
            manifest["provider"] = type(engine).__name__
        if schema is not None:
            if isinstance(schema, dict):
                manifest["schema"] = {
                    key: {"class_name": s.__name__,
                          "json_schema": s.model_json_schema()}
                    for key, s in schema.items()
                }
            else:
                manifest["schema"] = {
                    "class_name": schema.__name__,
                    "json_schema": schema.model_json_schema(),
                }
        finalize_manifest(out, manifest)
        _logger.info("Done. %d/%d succeeded. Manifest at %s",
                     manifest["n_articles_succeeded"],
                     manifest["n_articles_total"],
                     out / "manifest.json")
        return out

    # distill path — single prompt only (validated above)
    single_prompt = prompts[0]
    work = Path(work_dir).resolve() if work_dir else out / ".finefacts_work"
    work.mkdir(parents=True, exist_ok=True)
    gold_dir = work / "gold"

    _logger.info("Stage 1/5: gold gen via %s (n_workers=%d)", model, n_workers)
    extract_paid(corpus, [single_prompt], gold_dir, model=model,
                 max_article_chars=max_article_chars, limit=gold_size,
                 cache_dir=cache_dir, n_workers=n_workers, schema=schema,
                 max_input_tokens=max_input_tokens)

    corpus_idx = index_corpus_text(corpus, limit=gold_size)
    _logger.info("Stage 2/5: building training conversations")
    convs = work / "conversations.jsonl"
    n = build_conversations(gold_dir, single_prompt, max_article_chars, convs, corpus_idx)
    if n < 10:
        raise RuntimeError(f"Only {n} usable gold extractions; need ≥10 for training")
    train_p, dev_p = split_jsonl(convs, work / "training_data")
    _logger.info("  %d conversations → train + dev", n)

    _logger.info("Stage 3/5: LoRA training %s", distill_model)
    model_dir = work / "model"
    run_script("train.py",
            "--model_name", distill_model,
            "--train_file", train_p, "--dev_file", dev_p,
            "--output_dir", model_dir)

    _logger.info("Stage 4/5: merging adapter")
    merged = work / "merged"
    run_script("merge_adapter.py",
            "--adapter_dir", model_dir / "final",
            "--base_model", distill_model,
            "--output_dir", merged)

    _logger.info("Stage 5/5: distilled inference on rest")
    gold_ids = {p.stem for p in gold_dir.glob("*.json")}
    run_distilled(merged, corpus, single_prompt, out, gold_ids, max_article_chars)

    for p in gold_dir.glob("*.json"):
        shutil.copy2(p, out / p.name)
    finalize_manifest(out, manifest)
    _logger.info("Done. %d/%d succeeded. Manifest at %s",
                 manifest["n_articles_succeeded"],
                 manifest["n_articles_total"],
                 out / "manifest.json")
    return out
