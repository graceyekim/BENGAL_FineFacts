"""Distillation pipeline — build training conversations, train via subprocess,
merge LoRA adapter, run distilled inference on the rest of the corpus.

The training scripts ship inside the package at `finefacts/package/_scripts/`
and are invoked via subprocess so their argparse interface stays the source
of truth and their heavy deps (transformers/trl/peft/torch) don't load
until the user actually runs a distill.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from .corpus import iter_corpus
from .log import get_logger
from .parsing import article_id, article_meta, parse_json

_logger = get_logger(__name__)


_SCRIPTS = Path(__file__).resolve().parent / "_scripts"


def build_conversations(gold_dir, system_prompt, max_article_chars, out_path, corpus_idx):
    """Emit a JSONL of {conversations:[...]} training examples from a gold dir."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for jf in sorted(Path(gold_dir).glob("*.json")):
            rec = json.loads(jf.read_text(encoding="utf-8"))
            ext = rec.get("extracted")
            if not ext:
                continue
            aid = rec.get("article_id", jf.stem)
            text = corpus_idx.get(aid, "")[:max_article_chars]
            if not text:
                continue
            f.write(json.dumps({"conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {rec['title']}\n\n{text}"},
                {"role": "assistant", "content": json.dumps(ext, ensure_ascii=False)},
            ]}, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_jsonl(in_path, out_dir, dev_frac=0.1, seed=42):
    """Shuffle + split a JSONL file into train + dev under `out_dir`."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines = Path(in_path).read_text(encoding="utf-8").splitlines()
    random.Random(seed).shuffle(lines)
    n = max(1, int(len(lines) * dev_frac))
    (out_dir / "dev.jsonl").write_text("\n".join(lines[:n]) + "\n", encoding="utf-8")
    (out_dir / "train.jsonl").write_text("\n".join(lines[n:]) + "\n", encoding="utf-8")
    return out_dir / "train.jsonl", out_dir / "dev.jsonl"


def run_script(script, *args, env=None):
    """Run a bundled training script (e.g. `train.py`) as a subprocess."""
    cmd = [sys.executable, str(_SCRIPTS / script), *map(str, args)]
    e = {**os.environ, **(env or {})}
    _logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, env=e, check=True)


def index_corpus_text(corpus, limit=None):
    """Build {article_id: text} for fast lookup when emitting training convs."""
    idx = {}
    for art in iter_corpus(corpus):
        text = art.get("text") or art.get("body") or art.get("content") or ""
        if not text:
            continue
        idx[article_id(art, text)] = text
        if limit and len(idx) >= limit:
            break
    return idx


def run_distilled(merged, corpus, prompt, output, skip_ids, max_article_chars):
    """Load a merged LoRA-fine-tuned model and run inference on the corpus."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _logger.info("Loading distilled model from %s", merged)
    tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        merged, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    ).eval()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    for art in iter_corpus(corpus):
        text = (art.get("text") or art.get("body") or art.get("content") or "")[:max_article_chars]
        if len(text) < 100:
            continue
        m = article_meta(art, text)
        if m["article_id"] in skip_ids:
            continue
        path = out / f"{m['article_id']}.json"
        if path.exists():
            continue
        msgs = [{"role": "system", "content": prompt},
                {"role": "user", "content": f"Title: {m['title']}\n\n{text}"}]
        # Newer transformers' apply_chat_template returns a tensor directly only when
        # return_tensors is set AND return_dict is False; pinning return_dict=True is
        # the stable form across versions.
        enc = tok.apply_chat_template(
            msgs, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        ).to(mdl.device)
        with torch.no_grad():
            outs = mdl.generate(**enc, max_new_tokens=4096, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(outs[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        ext = parse_json(raw)
        rec = {**m, "extraction_date": date.today().isoformat(),
               "model": str(merged), "extracted": ext}
        if ext is None:
            rec["error"] = "json_parse_error"
            rec["raw_response"] = raw[:5000]
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
