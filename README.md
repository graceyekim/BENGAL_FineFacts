# FineFacts

Structured fact extraction from a text corpus via an LLM.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Point it at a corpus, write a prompt, get one JSON file per article plus a
manifest recording how the run was made. Runs against Claude / GPT / Gemini
via LiteLLM. Bring your own local engine by passing any object with a
`.generate()` method.

```python
import finefacts as ff

ff.extract(
    corpus="./articles/*.jsonl.gz",
    prompt="Extract every factual claim as JSON.",
    output="./out/",
)
```

```
./out/
├── manifest.json              # reproducibility receipt
├── run.log                    # API keys redacted
├── art_001.json               # { article_id, title, ..., extracted: { ... } }
├── art_002.json
└── ...
```

---

## Install

```bash
pip install -e .
```

Optional extras — pick what you need:

| Extra | Adds |
|---|---|
| `pip install -e ".[parquet]"` | Parquet corpus support |
| `pip install -e ".[distill]"` | LoRA fine-tuning (needs a GPU) |
| `pip install -e ".[analysis]"` | Embedding · clustering · semantic search |

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY / GEMINI_API_KEY / ...
```

---

## Quickstart

```python
import finefacts as ff

ff.extract(
    corpus=...,                        # csv / jsonl / jsonl.gz / parquet / list[dict]
    prompt="Extract claims as JSON: {claims: [...]}.",
    output="./out/",
    model="claude-sonnet-4-6",         # or gpt-4o, gemini/gemini-2.5-flash, ...
    n_workers=8,                       # 8 articles in flight at once
    max_spend=5.00,                    # $5 USD hard ceiling
    name="my-run",
)
```

Inspect the run from the CLI or from Python:

```bash
finefacts show ./out/
finefacts diff ./out_a/ ./out_b/
finefacts list ./runs/
```

```python
ff.show("./out/")              # → dict
ff.list_runs("./runs/")        # → list of summaries, newest first
ff.retry_failed("./out/")      # delete error records; rerun to re-attempt
```

---

## Three extraction modes

The shape of `prompt=` selects the mode:

```python
ff.extract(corpus, prompt="...", output=...)                       # single
ff.extract(corpus, prompt=[stage1, stage2, stage3], output=...)    # chained
ff.extract(corpus, prompt={"facts": p1, "tone": p2}, output=...)   # parallel
```

| Mode | When | Output shape |
|---|---|---|
| **Single** (`str`) | One prompt, one extraction | `extracted: { ... }` |
| **Chained** (`list[str]`) | Multi-stage: stage N sees stage N-1's output | `extracted: { ...final... }` + `_chain_stages: [...]` |
| **Parallel** (`dict[str, str]`) | Independent prompts on the same article | `extracted: { key1: {...}, key2: {...} }` + `_parallel_keys: [...]` |

---

## Pre-filter your corpus

Cheap gate before expensive extraction. When only a subset of your corpus is
relevant, filter first — filtered-out articles never trigger an LLM call.

**The contract:** `filter=` accepts ANY callable that takes an article dict
and returns `bool` (True = keep, False = skip). Write your own, use a
built-in helper, or compose several together.

### Option 1 — your own function

Write any Python function; pass it as `filter=`.

```python
def my_filter(art):
    """Keep articles that are (a) longer than 500 chars, (b) from state media,
    and (c) mention a country of interest."""
    text = art.get("text", "")
    if len(text) < 500:
        return False
    if art.get("source_domain") not in {"rt.com", "tass.com", "presstv.ir"}:
        return False
    return any(c in text.lower() for c in ("russia", "china", "iran"))

ff.extract(corpus=..., prompt=..., output="./out/", filter=my_filter)
```

Lambdas, regexes, external ML models, database lookups — anything that
returns True/False for an article dict works.

```python
import re
russia_re = re.compile(r"\b(russia|ukraine|crimea|donbas)\b", re.I)
ff.extract(corpus=..., prompt=..., output=...,
           filter=lambda art: bool(russia_re.search(art.get("text", ""))))
```

### Option 2 — `ff.keyword_filter` (substring, $0 cost)

```python
ff.extract(
    corpus=all_articles,
    prompt=extract_prompt,
    output="./out/",
    filter=ff.keyword_filter(["Russia", "Ukraine", "Crimea"]),   # match="any" default
)
```

### Option 3 — `ff.llm_filter` (cheap LLM classifier)

Cheap-gate pattern: a Haiku call decides YES/NO before the expensive Sonnet
extraction runs. Order-of-magnitude cost savings on broad corpora where only
a fraction is on-topic. Verdicts are cached, so a rerun with the same
classifier is free.

```python
classifier = ff.llm_filter(
    "Is this article primarily about the Russia-Ukraine war? Respond YES or NO.",
    model="claude-haiku-4-5",
)
ff.extract(corpus=all_articles, prompt=extract_prompt, output="./out/",
           filter=classifier)
```

### Option 4 — combine filters with `ff.compose_filters`

Any mix — your own function, keyword, LLM classifier — combined by AND or OR.

```python
combined = ff.compose_filters(
    lambda art: art.get("source_domain") in STATE_MEDIA_DOMAINS,   # must be state media
    ff.keyword_filter(["Russia", "Ukraine"]),                      # must mention topic
    ff.llm_filter("Is this news content? YES/NO.",
                  model="claude-haiku-4-5"),                       # must be news (not sports)
    mode="all",                                                    # AND — all three
)
ff.extract(corpus=..., prompt=..., output=..., filter=combined)
```

Manifest records `filter.applied`, `filter.name`, and `filter.n_skipped` so
you can see how much the pre-filter saved you.

## Schema enforcement (Pydantic)

```python
from pydantic import BaseModel
from typing import List

class Claim(BaseModel):
    text: str
    confidence: float

class Extraction(BaseModel):
    claims: List[Claim]

ff.extract(corpus=..., prompt="Extract claims with confidence.",
           output="./out/", schema=Extraction)
```

Routed through the provider's native structured-output API (OpenAI
`response_format`, Anthropic `tool_use`) and validated locally. Validation
failures keep the raw parsed output under `extracted` and flag the record
with an `error` field — nothing is lost.

For per-key parallel schemas:

```python
ff.extract(
    corpus=...,
    prompt={"facts": fact_prompt, "sentiment": sentiment_prompt},
    schema={"facts": Facts, "sentiment": Sentiment},
    output="./out/",
)
```

---

## Multilingual

Two strategies for cross-language corpora:

```python
# Direct: 1 LLM call per article, output forced to English.
ff.extract(corpus=articles, prompt="...", output="./out/",
           output_language="en")

# Two-stage: pre-translate, then extract; translation saved as an artifact.
ff.extract(corpus=articles, prompt="...", output="./out/",
           output_language="en", translate_first=True)
```

Accepts ISO 639-1 codes (`"en"`, `"ko"`, `"zh"`, `"ja"`, …) or English names.

---

## Per-article templates

Sprinkle `{{ var }}` in the prompt; values come from each article's dict:

```python
ff.extract(
    corpus=[
        {"id": "a1", "text": "...", "region": "Korea", "year": 2024},
        {"id": "a2", "text": "...", "region": "Brazil", "year": 2023},
    ],
    prompt="Extract claims from {{ region }} articles published in {{ year }}.",
    output="./out/",
)
```

---

## Non-standard corpora

Alias your columns:

```python
ff.extract(
    corpus=hf_dataset,
    prompt=...,
    output="./out/",
    fields={"text": "body", "id": "post_id", "title": "headline"},
)
```

---

## Cost-aware

```python
# Dry-run preview (no API spend)
est = ff.estimate_cost(corpus=..., prompt=..., model="claude-sonnet-4-6")
# → {"n_articles": 50, "cost_usd": 3.09, "per_article_usd": 0.062, ...}

# Real run with a hard ceiling
ff.extract(corpus=..., prompt=..., output="./out/", max_spend=10.00)
```

Pricing comes from [LiteLLM](https://litellm.ai)'s `completion_cost`,
auto-updating for new model releases. Cache hits cost $0. The manifest
records the actual total once the run completes.

---

## Bring your own engine

By default `ff.extract` uses LiteLLM for hosted providers (Claude, GPT,
Gemini). To run a local model, pass any object with a
`.generate(model, system, user, *, schema, tracker) -> str` method:

```python
class MyLocalEngine:
    def generate(self, model, system, user, *, schema=None, tracker=None):
        # your local inference here
        return "..."

ff.extract(..., provider=MyLocalEngine(), model="my-model-id")
```

Bring-your-own engines ignore `schema=` (no provider-native structured output
— use Pydantic post-validation instead) and `tracker=` / `max_spend=` (no
API cost to track).

## Fine-tune on existing gold

Once `ff.extract` has produced gold JSONs, you can distill a small OSS model
on that gold WITHOUT re-paying for gold generation. Useful when iterating
on training hyperparameters or comparing base models.

```python
# Step 1: generate gold once (paid)
ff.extract(corpus="./articles.jsonl", prompt=my_prompt,
           output="./gold/", model="claude-sonnet-4-6")

# Step 2: fine-tune on that gold (unlimited iteration, no API cost)
ff.train(
    gold="./gold/",
    corpus="./articles.jsonl",
    system_prompt=my_prompt,
    output="./trained/",
    base_model="Qwen/Qwen3-4B",
)

# Step 3: use the trained model for future extraction
ff.extract(corpus=..., prompt=..., output="./out/",
           provider=my_local_engine, model="./trained/")
```

This is the standalone version of `ff.extract(..., distill=True)`: same
training pipeline, but decoupled from gold-gen so you can swap base models
or reuse the same gold across many training runs. SFT only for now —
add DPO / ORPO later only if a real failure mode emerges.

## Distill for scale

When the corpus outgrows the API budget, fine-tune a small open model on a
sample:

```python
ff.extract(
    corpus=...,
    prompt=...,
    output="./out/",
    distill=True,
    distill_model="Qwen/Qwen3-4B",
    gold_size=5000,                    # articles for the teacher gold set
)
```

Five stages: gold-gen → conversation build → LoRA SFT → adapter merge →
local inference on the rest. Requires a CUDA GPU.

---

## Downstream analysis

```bash
pip install -e ".[analysis]"
```

```python
ff.embed("./out/", granularity="claim")          # sentence-transformers + sqlite-vec
ff.cluster("./out/", algorithm="hdbscan")        # cluster IDs written to the index
ff.detect_syndication("./out/", similarity_threshold=0.92)
hits = ff.search("./out/", "climate finance", k=10)
```

Vector index + metadata live in a single `output/index.sqlite` joinable to
the per-article JSONs by `article_id`. Heavy deps lazy-load — `import
finefacts` is free even without the `[analysis]` extra.

---

## Optimize a prompt (AutoRubric)

```python
best = ff.optimize_prompt(
    initial_prompt=open("my_prompt.txt").read(),
    sample_corpus=articles[:25],
    iterations=7,
    judge_model="claude-sonnet-4-6",
    target_model="claude-sonnet-4-6",
)
# → {"best_prompt": "...", "best_score": 0.677, "history": [...]}
```

Each iteration extracts on the sample, judges against a weighted rubric
(bundled generic or your own YAML), picks the weakest criterion, and asks
an improver model to revise the prompt. All intermediate prompts and judge
verdicts are written to disk.

---

## Quick API reference

```python
ff.extract(corpus, prompt, output, *,
           model="claude-sonnet-4-6",
           provider=None,                  # None / "litellm" (default), or an engine object with .generate()
           n_workers=1,                    # thread-pool size
           schema=None,                    # Pydantic BaseModel or dict[str, BaseModel]
           output_language=None,           # force output language
           translate_first=False,          # 2-stage with translation artifact
           max_spend=None,                 # USD ceiling
           fields=None,                    # column-name aliases
           filter=None,                    # Callable[[dict], bool] pre-filter
           max_input_tokens=None,          # auto-detected from model
           capture_confidence=False,       # hoist self-reported confidence
           batch=False,                    # Anthropic Message Batches API (50% off)
           batch_poll_interval=60.0, batch_timeout=None,
           progress=None,                  # None=auto, True/False force
           confirm=True, cache=True,       # interactive y/N + persistent cache
           distill=False, distill_model=None, gold_size=1000,
           name=None, verbose=False, quiet=False)
# → Path (the output directory)

ff.optimize_prompt(...)
ff.evaluate(output, corpus, rubric=None, *, judge_model, sample_size=None)
ff.compare_by_rubric(dir_a, dir_b, corpus, rubric=None, *, judge_model, tolerance=1.0)
ff.load_rubric(path_or_None)              # → list of criterion dicts
ff.estimate_cost(corpus, prompt, *, mode="extract" | "distill" | "optimize", **kw)

ff.train(gold, corpus, system_prompt, output, *,
         base_model="Qwen/Qwen3-4B", method="sft", ...)

ff.keyword_filter(keywords, *, field="text", match="any", case_sensitive=False)
ff.llm_filter(classifier_prompt, *, model="claude-haiku-4-5", positive_pattern=r"\byes\b")
ff.compose_filters(*filters, mode="all" | "any")

ff.load(out_dir, include_errors=True)      # → list[dict] of per-article records
ff.show(out_dir)                           # → dict {manifest, sample_extraction}
ff.list_runs(parent_dir)                   # → list of summary dicts, newest first
ff.compare_runs(dir_a, dir_b, *, match=None)   # → per-article diff + rates
ff.match_by_field(field, *, case_sensitive=False)
ff.match_by_keywords(keywords, *, field, match="any")
ff.match_by_regex(pattern, *, field)
ff.retry_failed(out_dir)                   # → {n_deleted, deleted_ids}
ff.to_csv(out_dir, out_path=None)          # flatten nested dicts, one row per article
ff.to_jsonl(out_dir, out_path=None)        # preserve structure, one line per article

ff.embed(out_dir, granularity="claim" | "article", model=..., rerun=False)
ff.cluster(out_dir, algorithm="hdbscan" | "kmeans", ...)
ff.detect_syndication(out_dir, similarity_threshold, min_shared_claims, top_k)
ff.search(out_dir, query, k=10, article_id=None)
```

CLI:

```
finefacts show <out> [--json]
finefacts diff <a> <b>
finefacts list <parent> [--json]
```

---

## Output schema

Per-article JSON at `output/{article_id}.json`:

```json
{
  "article_id": "...",
  "title": "...",
  "source_url": "...",
  "source_domain": "...",
  "extraction_date": "2026-06-30",
  "model": "claude-sonnet-4-6",
  "extracted": { /* whatever your prompt asked for */ }
}
```

Mode-specific fields:

| Mode | Extra fields |
|---|---|
| Chained | `_chain_stages: [{prompt_idx, raw}, ...]` |
| Parallel | `_parallel_keys: [...]`, `_parallel_errors: {...}` on failures |
| `translate_first` | `translated_to: "en"`, `translation: "..."` |
| Schema validation fail | `error: "schema_validation_error: ..."`, raw parsed still kept |

Every run dir also has a top-level `manifest.json` with the prompt hash,
model, counts, schema, cost tracking, and timestamps. Inspect with
`finefacts show <out>`.

---

## Running anywhere

| Path | What's needed |
|---|---|
| **Paid-API extraction** | Python 3.10+ and an API key. Runs anywhere. |
| **+ Distillation** | + a CUDA GPU. Mac (MPS), workstation, Colab, cloud, HPC. |
| **+ Analysis** | + [analysis] extras (sentence-transformers, sqlite-vec, hdbscan). |

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
