"""Embed extracted content from a finefacts run dir.

Default backend: sentence-transformers (`all-MiniLM-L6-v2`, 384-d). Vectors and
metadata are stored together in `output/index.sqlite` via sqlite-vec, joinable
to the per-article JSON outputs through `article_id`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..log import get_logger

_logger = get_logger(__name__)


_ANALYSIS_EXTRAS_HINT = (
    "ff.embed requires the [analysis] extras. Run: pip install finefacts[analysis]"
)


def _flatten_claims(extracted):
    """Best-effort extraction of individual claim strings from schema-agnostic JSON.

    Heuristic, in order:
      1. `extracted` is a list of dicts/strings → one row per item.
      2. `extracted` is a dict with a known "claims-like" top-level list field
         (claims, facts, atomic_claims, decontextualized_claims, items) → use that.
      3. Otherwise: walk the structure and yield any string longer than 20 chars.
    """
    if extracted is None:
        return []
    if isinstance(extracted, str):
        return [extracted] if len(extracted) > 20 else []
    if isinstance(extracted, list):
        out = []
        for item in extracted:
            if isinstance(item, str) and len(item) > 20:
                out.append(item)
            elif isinstance(item, dict):
                out.append(json.dumps(item, ensure_ascii=False))
        return out
    if isinstance(extracted, dict):
        for key in ("claims", "facts", "atomic_claims", "decontextualized_claims",
                    "extracted_claims", "claim", "items"):
            v = extracted.get(key)
            if isinstance(v, list):
                return _flatten_claims(v)

        out = []

        def walk(v):
            if isinstance(v, str) and len(v) > 20:
                out.append(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    walk(vv)
            elif isinstance(v, list):
                for vv in v:
                    walk(vv)

        walk(extracted)
        return out
    return []


def _flatten_article_text(rec):
    """Article-level text = title + flattened claims joined."""
    title = rec.get("title", "") or ""
    parts = [title] + _flatten_claims(rec.get("extracted"))
    return "\n".join(p for p in parts if p)[:8000]


def _open_db(db_path: Path, dim: int):
    import sqlite3
    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(embedding float[{dim}])"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS items (
               rowid INTEGER PRIMARY KEY,
               item_id TEXT UNIQUE,
               article_id TEXT,
               text TEXT,
               cluster_id INTEGER
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS items_article_id ON items(article_id)"
    )
    return conn


def embed(output, *, granularity: str = "claim",
          model: str = "all-MiniLM-L6-v2",
          rerun: bool = False,
          batch_size: int = 64):
    """Build a vector index from a finefacts run dir.

    Args:
        output:        path to an existing finefacts run dir (has `manifest.json`).
        granularity:   "claim" (one row per extracted fact, default) or "article"
                       (one row per article, text = title + flattened claims).
        model:         sentence-transformers model id; default is fast 384-d.
        rerun:         if False (default) and `analysis.embed` already in the manifest,
                       skip with a log message.
        batch_size:    encoder batch size.

    Returns the `analysis.embed` block written to manifest.json.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import sqlite_vec  # noqa: F401  (load extension via _open_db)
    except ImportError as e:
        raise ImportError(_ANALYSIS_EXTRAS_HINT) from e

    if granularity not in ("claim", "article"):
        raise ValueError(f"granularity must be 'claim' or 'article'; got {granularity!r}")

    out = Path(output).resolve()
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json at {manifest_path}. Run ff.extract first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not rerun and manifest.get("analysis", {}).get("embed"):
        prev = manifest["analysis"]["embed"]
        _logger.info("embed: already embedded (%s vectors); pass rerun=True to recompute",
                     prev.get("n_vectors"))
        return prev

    # Collect (item_id, article_id, text) rows.
    rows = []
    for jf in sorted(out.glob("*.json")):
        if jf.name == "manifest.json":
            continue
        try:
            rec = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "error" in rec or rec.get("_parallel_errors"):
            continue
        aid = rec.get("article_id") or jf.stem
        if granularity == "article":
            text = _flatten_article_text(rec)
            if len(text) >= 20:
                rows.append((aid, aid, text))
        else:
            for i, claim in enumerate(_flatten_claims(rec.get("extracted"))):
                rows.append((f"{aid}#{i}", aid, claim))

    if not rows:
        raise RuntimeError(
            f"No usable {granularity}-level rows found in {out}. "
            "Check that ff.extract produced non-empty extractions."
        )

    _logger.info("embed: loading model %s", model)
    enc = SentenceTransformer(model)
    dim = enc.get_sentence_embedding_dimension()

    db_path = out / "index.sqlite"
    conn = _open_db(db_path, dim)
    try:
        total = len(rows)
        n_inserted = 0
        for start in range(0, total, batch_size):
            batch = rows[start:start + batch_size]
            texts = [r[2] for r in batch]
            vecs = enc.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            for (item_id, art_id, text), v in zip(batch, vecs):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO items (item_id, article_id, text) VALUES (?, ?, ?)",
                    (item_id, art_id, text),
                )
                if cur.rowcount == 0:
                    continue  # already indexed; resume
                rowid = cur.lastrowid
                conn.execute(
                    "INSERT INTO vecs (rowid, embedding) VALUES (?, ?)",
                    (rowid, np.asarray(v, dtype=np.float32).tobytes()),
                )
                n_inserted += 1
            conn.commit()
            _logger.info("  embed %d/%d", min(start + batch_size, total), total)
    finally:
        conn.close()

    info = {
        "backend": "sentence-transformers",
        "model": model,
        "granularity": granularity,
        "dim": dim,
        "n_vectors": n_inserted,
        "n_rows_seen": len(rows),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest.setdefault("analysis", {})["embed"] = info
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    _logger.info("embed done. %d new vectors @ %d-d → %s", n_inserted, dim, db_path)
    return info
