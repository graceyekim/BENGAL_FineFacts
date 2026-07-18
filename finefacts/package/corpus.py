"""Corpus iteration — load articles from CSV / JSONL / JSONL.gz / Parquet / iterables.

Each article is a `dict`. Required key: `text` (or `body` or `content`).
Optional keys: `title`, `id`/`article_id`, `url`/`source_url`, `source_domain`/`base_url`.
"""

from __future__ import annotations

import csv
import glob as _glob
import gzip
import json
import sys
from pathlib import Path


def iter_corpus(corpus):
    """corpus → iterable of dicts. Accepts iterable, file path, or glob string."""
    if not isinstance(corpus, (str, Path)):
        yield from corpus
        return
    csv.field_size_limit(sys.maxsize)
    for path in sorted(_glob.glob(str(corpus))) or [str(corpus)]:
        s = str(path).lower()
        if s.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif s.endswith(".jsonl"):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif s.endswith(".csv"):
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                yield from csv.DictReader(f)
        elif s.endswith(".parquet"):
            import pyarrow.parquet as pq
            yield from pq.read_table(path).to_pylist()
        else:
            raise ValueError(f"Unknown corpus format: {path}")
