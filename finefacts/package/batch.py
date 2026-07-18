"""Anthropic Message Batches integration for `ff.extract(batch=True)`.

Single-shot prompts only; cache is bypassed (the batch is the resumability
mechanism). Anthropic models only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

from .log import get_logger
from .parsing import article_meta, parse_json
from .template import has_template, render

_logger = get_logger(__name__)


_ANTHROPIC_HINT = (
    "Batch API requires the `anthropic` package (already in base deps). "
    "If you see this, run `pip install -e .` to install."
)


@dataclass
class BatchRequest:
    """One item in a batch — an article's LLM request."""
    custom_id: str
    system: str
    user: str
    max_tokens: int = 8192
    temperature: float = 0.0


@dataclass
class BatchResult:
    """Per-item outcome after batch processing."""
    custom_id: str
    ok: bool
    content: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class BatchStatus:
    """Live snapshot of a batch mid-flight."""
    id: str
    status: str  # "in_progress" | "ended" | "canceling" | ...
    processing_count: int = 0
    succeeded_count: int = 0
    errored_count: int = 0
    total_count: int = 0
    created_at: str = ""


class AnthropicBatchRunner:
    """Thin wrapper around anthropic.Anthropic().messages.batches.

    Kept as a class so tests can stub it in without touching the anthropic
    SDK. Instantiate directly for real runs; pass a duck-typed replacement
    to `run_batch(runner=...)` for tests.
    """

    def __init__(self):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(_ANTHROPIC_HINT) from e
        self._client = anthropic.Anthropic()

    def submit(self, model: str, requests: list[BatchRequest]) -> str:
        """Submit a batch; return the Anthropic-assigned batch id."""
        body = [
            {
                "custom_id": r.custom_id,
                "params": {
                    "model": model,
                    "max_tokens": r.max_tokens,
                    "temperature": r.temperature,
                    "messages": [
                        {"role": "user", "content": r.user},
                    ],
                    "system": r.system,
                },
            }
            for r in requests
        ]
        batch = self._client.messages.batches.create(requests=body)
        return batch.id

    def poll(self, batch_id: str) -> BatchStatus:
        """Fetch current status of an in-flight batch."""
        b = self._client.messages.batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None) or {}
        # anthropic SDK returns a pydantic-ish object; access as attribute or key
        def _g(k, default=0):
            if hasattr(counts, k):
                return getattr(counts, k) or default
            if isinstance(counts, dict):
                return counts.get(k, default)
            return default
        return BatchStatus(
            id=batch_id,
            status=b.processing_status,
            processing_count=_g("processing"),
            succeeded_count=_g("succeeded"),
            errored_count=_g("errored"),
            total_count=(_g("processing") + _g("succeeded")
                        + _g("errored") + _g("canceled") + _g("expired")),
            created_at=str(getattr(b, "created_at", "")),
        )

    def results(self, batch_id: str) -> Iterator[BatchResult]:
        """Stream per-item results from a completed batch."""
        for entry in self._client.messages.batches.results(batch_id):
            cid = entry.custom_id
            r = entry.result
            if r.type == "succeeded":
                # message.content is a list of blocks; take the first text one
                try:
                    text = next(
                        (b.text for b in r.message.content if hasattr(b, "text")),
                        "",
                    )
                except Exception:
                    text = str(r.message)
                usage = getattr(r.message, "usage", None)
                in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                yield BatchResult(custom_id=cid, ok=True, content=text,
                                   input_tokens=in_tok, output_tokens=out_tok)
            else:
                err_msg = getattr(r, "error", None) or f"batch entry {r.type}"
                yield BatchResult(custom_id=cid, ok=False,
                                   error=f"{r.type}: {err_msg}")


def _wait_for_completion(runner, batch_id: str, *, poll_interval: float,
                         timeout: float | None, progress=None):
    """Block until batch reaches a terminal status; yield status snapshots."""
    start = time.monotonic()

    pbar = None
    if progress is not False:
        try:
            from tqdm import tqdm
            pbar = tqdm(desc="batch", unit="art", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        while True:
            status = runner.poll(batch_id)
            done = (status.succeeded_count + status.errored_count)
            if pbar is not None:
                if status.total_count and pbar.total != status.total_count:
                    pbar.total = status.total_count
                    pbar.refresh()
                pbar.n = done
                pbar.set_postfix({"status": status.status})
                pbar.refresh()
            _logger.info("batch %s: status=%s done=%d/%d",
                         batch_id, status.status, done, status.total_count)
            if status.status in ("ended", "canceled", "expired", "failed"):
                return status
            if timeout is not None and (time.monotonic() - start) > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {timeout}s "
                    f"(status: {status.status})"
                )
            time.sleep(poll_interval)
    finally:
        if pbar is not None:
            pbar.close()


def run_batch(
    corpus_iter,
    prompt: str,
    output,
    *,
    model: str,
    max_article_chars: int,
    max_input_tokens=None,
    fields=None,
    filter_fn=None,
    filter_skip_counter=None,
    runner=None,
    poll_interval: float = 60.0,
    timeout: float | None = None,
    progress=None,
    tracker=None,
) -> BatchStatus:
    """Run a batch extraction end-to-end.

    Applies field-aliasing, filter, and token-budget truncation exactly like
    the normal extract path — then bundles every article into ONE Anthropic
    batch, polls until done, retrieves results, writes per-article JSONs.

    Returns the final BatchStatus.
    """
    from .extractor import _alias_iter, _filter_iter, _truncate_article

    out = Path(output); out.mkdir(parents=True, exist_ok=True)

    if runner is None:
        runner = AnthropicBatchRunner()

    skip_counter = filter_skip_counter if filter_skip_counter is not None else [0]
    aliased = _alias_iter(corpus_iter, fields)
    filtered = _filter_iter(aliased, filter_fn, skip_counter)

    # Assemble batch requests
    requests: list[BatchRequest] = []
    id_to_meta: dict[str, dict] = {}
    for art in filtered:
        text = _truncate_article(art, model, [prompt],
                                 max_article_chars, max_input_tokens)
        if len(text) < 100:
            continue
        m = article_meta(art, text)
        path = out / f"{m['article_id']}.json"
        if path.exists():
            continue  # honor resumability
        user = f"Title: {m['title']}\n\n{text}"
        # Per-article template substitution — matches chained/parallel paths.
        rendered_prompt = render(prompt, art) if has_template(prompt) else prompt
        requests.append(BatchRequest(
            custom_id=m["article_id"], system=rendered_prompt, user=user,
        ))
        id_to_meta[m["article_id"]] = m

    if not requests:
        _logger.info("batch: nothing to submit (0 new articles)")
        return BatchStatus(id="", status="ended", total_count=0)

    _logger.info("batch: submitting %d requests (model=%s)", len(requests), model)
    batch_id = runner.submit(model, requests)
    _logger.info("batch: submitted id=%s", batch_id)

    status = _wait_for_completion(runner, batch_id,
                                   poll_interval=poll_interval,
                                   timeout=timeout, progress=progress)

    if status.status != "ended":
        raise RuntimeError(
            f"Batch {batch_id} finished in status={status.status!r}; "
            f"succeeded={status.succeeded_count} errored={status.errored_count}"
        )

    # Retrieve + write per-article JSONs. Feed the cost tracker as we go —
    # batch pricing is 50% of the standard per-token rate on both input+output.
    n_written = 0
    for res in runner.results(batch_id):
        m = id_to_meta.get(res.custom_id)
        if not m:
            continue
        path = out / f"{res.custom_id}.json"
        if not res.ok:
            rec = {**m, "extraction_date": date.today().isoformat(),
                   "model": model, "extracted": None,
                   "error": res.error, "raw_response": ""}
        else:
            parsed = parse_json(res.content)
            rec = {**m, "extraction_date": date.today().isoformat(),
                   "model": model, "extracted": parsed}
            if parsed is None:
                rec["error"] = "json_parse_error"
                rec["raw_response"] = res.content[:5000]
            if tracker is not None:
                _feed_tracker(tracker, model, res.input_tokens, res.output_tokens)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        n_written += 1

    _logger.info("batch done. wrote %d records to %s", n_written, out)
    return status


def _feed_tracker(tracker, model: str, input_tokens: int, output_tokens: int) -> None:
    """Record a batch call's cost against the tracker at the 50% batch rate.

    Best-effort: if LiteLLM has no price entry for the model, record $0 —
    matches `llms.call_llm`'s behavior on unpriced models.
    """
    try:
        import litellm
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        tracker.add((prompt_cost + completion_cost) * 0.5)
    except Exception:
        tracker.add(0.0)
