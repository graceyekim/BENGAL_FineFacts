"""Nearest-neighbor search over embedded claims at query time."""

from __future__ import annotations

import json
from pathlib import Path


_ANALYSIS_EXTRAS_HINT = (
    "ff.search requires the [analysis] extras. Run: pip install finefacts[analysis]"
)


def search(output, query: str, *, k: int = 10,
           model: str | None = None,
           article_id: str | None = None):
    """Search the vector index for claims similar to `query`.

    Args:
        output:     path to a finefacts run dir (must have embeddings).
        query:      natural-language query.
        k:          number of results to return.
        model:      sentence-transformers model id; default matches the model
                    used at embedding time (read from manifest).
        article_id: if set, restrict matches to a single article.

    Returns a list of dicts:
        {item_id, article_id, text, similarity}
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import sqlite_vec  # noqa: F401
    except ImportError as e:
        raise ImportError(_ANALYSIS_EXTRAS_HINT) from e

    out = Path(output).resolve()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embed_block = manifest.get("analysis", {}).get("embed")
    if not embed_block:
        raise RuntimeError(
            f"No embed block in {manifest_path}. Run ff.embed first."
        )

    effective_model = model or embed_block["model"]
    enc = SentenceTransformer(effective_model)
    q = enc.encode([query], normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False)[0]
    q_bytes = np.asarray(q, dtype=np.float32).tobytes()

    import sqlite3
    conn = sqlite3.connect(out / "index.sqlite")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        if article_id is None:
            rows = conn.execute(
                "SELECT items.item_id, items.article_id, items.text, vecs.distance "
                "FROM vecs JOIN items ON items.rowid = vecs.rowid "
                "WHERE vecs.embedding MATCH ? "
                "ORDER BY vecs.distance LIMIT ?",
                (q_bytes, k),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT items.item_id, items.article_id, items.text, vecs.distance "
                "FROM vecs JOIN items ON items.rowid = vecs.rowid "
                "WHERE vecs.embedding MATCH ? AND items.article_id = ? "
                "ORDER BY vecs.distance LIMIT ?",
                (q_bytes, article_id, k),
            ).fetchall()
    finally:
        conn.close()

    return [
        {
            "item_id": r[0],
            "article_id": r[1],
            "text": r[2],
            "similarity": 1.0 - (float(r[3]) ** 2) / 2.0,
        }
        for r in rows
    ]
