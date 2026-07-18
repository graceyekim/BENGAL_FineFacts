"""Standalone SFT — train a small OSS model on gold you already have.

This is the training-only slice of the `ff.extract(distill=True)` pipeline —
useful when you already have per-article gold from a prior extraction and
want to iterate on training hyperparameters, base models, or the training
data itself WITHOUT paying to re-generate gold every time.

Contract:
  input:  a directory of per-article gold JSONs (from `ff.extract`), the
          same source corpus, and the same system prompt used to generate them.
  output: a merged LoRA-fine-tuned model saved to `output/`.

DPO is intentionally not implemented — SFT is enough for structured
extraction tasks. If you develop a specific failure mode DPO can fix, we
add it then.
"""

from __future__ import annotations

from pathlib import Path

from .distill import (
    build_conversations,
    index_corpus_text,
    run_script,
    split_jsonl,
)
from .log import configure as _configure_log, get_logger

_logger = get_logger(__name__)


def train(
    gold,
    corpus,
    system_prompt: str,
    output,
    *,
    base_model: str = "Qwen/Qwen3-4B",
    method: str = "sft",
    max_article_chars: int = 6000,
    dev_frac: float = 0.1,
    work_dir=None,
    name: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> Path:
    """Fine-tune a small OSS model on existing gold extractions.

    Args:
        gold:              directory of per-article gold JSONs from a prior
                           `ff.extract` run (each with `article_id`, `title`,
                           `extracted` fields).
        corpus:            the source corpus used to generate the gold. Needed
                           to look up article text — the gold JSONs don't store
                           the full body. Accepts glob path or iterable of dicts.
        system_prompt:     the system prompt that was used to generate the gold.
                           Recorded on every training example so the fine-tuned
                           model sees the same prompt at inference time.
        output:            directory where the merged fine-tuned model will be
                           saved. Created if missing.
        base_model:        HuggingFace model id to fine-tune. Default:
                           `Qwen/Qwen3-4B`. Must be a causal LM.
        method:            `"sft"` — only mode supported. Reserved for future
                           `"dpo"` / `"orpo"` etc.
        max_article_chars: truncate articles at this many chars when building
                           training conversations.
        dev_frac:          fraction of gold reserved for dev set.
        work_dir:          scratch dir for intermediate artifacts (train.jsonl,
                           dev.jsonl, LoRA adapter). Defaults to
                           `output/.finefacts_train_work`.
        name:              human label for the run; appears in logs.
        verbose / quiet:   log level.

    Returns:
        `Path` to the merged model directory (same as `output`).

    Requires:
        - `pip install finefacts[distill]` (transformers, trl, peft, ...)
        - A CUDA GPU visible to the training subprocess.

    Example:
        # First: generate gold via ff.extract
        ff.extract(corpus="./articles.jsonl", prompt=my_prompt,
                   output="./gold/", model="claude-sonnet-4-6")

        # Then: fine-tune on that gold — no more API cost
        ff.train(
            gold="./gold/",
            corpus="./articles.jsonl",
            system_prompt=my_prompt,
            output="./trained/",
            base_model="Qwen/Qwen3-4B",
        )

        # Iterate: try a different base model on the same gold — cheap
        ff.train(gold="./gold/", corpus="./articles.jsonl",
                 system_prompt=my_prompt, output="./trained_llama/",
                 base_model="meta-llama/Llama-3.2-3B")
    """
    if method != "sft":
        raise NotImplementedError(
            f"method={method!r} not supported. Only 'sft' is available today. "
            "DPO / ORPO / other preference-based methods are deferred — open "
            "an issue with a concrete failure mode if you need them."
        )

    gold_dir = Path(gold)
    if not gold_dir.is_dir():
        raise FileNotFoundError(f"gold directory not found: {gold_dir}")

    n_gold = sum(1 for f in gold_dir.glob("*.json") if f.name != "manifest.json")
    if n_gold < 10:
        raise RuntimeError(
            f"Only {n_gold} gold files in {gold_dir}; need ≥10 for training. "
            f"Run `ff.extract` on more articles first."
        )

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")

    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir).resolve() if work_dir else out / ".finefacts_train_work"
    work.mkdir(parents=True, exist_ok=True)
    effective_name = name or out.name

    _configure_log(run_name=effective_name, verbose=verbose, quiet=quiet,
                   log_file=out / "train.log")
    _logger.info("train (%s): base_model=%s gold=%d", method, base_model, n_gold)

    _logger.info("Stage 1/4: indexing corpus")
    corpus_idx = index_corpus_text(corpus)
    if not corpus_idx:
        raise RuntimeError(
            f"Corpus produced 0 usable articles — cannot look up gold text. "
            f"Check that the corpus argument matches what `ff.extract` was run on."
        )

    _logger.info("Stage 2/4: building training conversations from %d gold files", n_gold)
    convs = work / "conversations.jsonl"
    n = build_conversations(gold_dir, system_prompt, max_article_chars, convs, corpus_idx)
    if n < 10:
        raise RuntimeError(
            f"Only {n} usable gold extractions after joining to corpus text; "
            f"need ≥10 for training. Common cause: gold article_id doesn't "
            f"match corpus article id — check your `fields=` mapping."
        )
    train_p, dev_p = split_jsonl(convs, work / "training_data", dev_frac=dev_frac)
    _logger.info("  %d conversations → train + dev", n)

    _logger.info("Stage 3/4: LoRA %s training %s", method.upper(), base_model)
    adapter_dir = work / "model"
    run_script(
        "train.py",
        "--model_name", base_model,
        "--train_file", train_p,
        "--dev_file", dev_p,
        "--output_dir", adapter_dir,
    )

    _logger.info("Stage 4/4: merging adapter → %s", out)
    run_script(
        "merge_adapter.py",
        "--adapter_dir", adapter_dir / "final",
        "--base_model", base_model,
        "--output_dir", out,
    )

    _logger.info("train done. Merged model at %s", out)
    return out
