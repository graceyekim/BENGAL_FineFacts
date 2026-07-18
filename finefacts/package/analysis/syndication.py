"""Detect article pairs with high cross-article claim similarity.

Approach: for every claim, ask the vector index for its top-K nearest
neighbors. If neighbor X belongs to a different article B than the source
claim's article A, that's a "shared claim" between A and B. Aggregate to
article-pair level and emit pairs where the shared-claim count crosses a
threshold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..log import get_logger

_logger = get_logger(__name__)


_ANALYSIS_EXTRAS_HINT = (
    "ff.detect_syndication requires the [analysis] extras. "
    "Run: pip install finefacts[analysis]"
)


def detect_syndication(output, *,
                        similarity_threshold: float = 0.92,
                        min_shared_claims: int = 3,
                        top_k: int = 5,
                        rerun: bool = False):
    """Find article pairs that share many highly-similar claims (syndication).

    Args:
        output:                path to a finefacts run dir (must have embeddings).
        similarity_threshold:  cosine sim to count two claims as "shared" (0–1).
        min_shared_claims:     minimum shared-claim count for a pair to be emitted.
        top_k:                 nearest neighbors per claim (small → cheap, may miss
                               distant matches; large → expensive).
        rerun:                 if False (default) and `analysis.syndication` already
                               present, skip and return the cached block.

    Writes `output/syndication.jsonl` with one record per pair:
        {"article_a", "article_b", "n_shared_claims", "mean_similarity", ...}
    """
    try:
        import numpy as np
        import sqlite_vec  # noqa: F401
    except ImportError as e:
        raise ImportError(_ANALYSIS_EXTRAS_HINT) from e

    out = Path(output).resolve()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "embed" not in manifest.get("analysis", {}):
        raise RuntimeError(
            f"No embed block in {manifest_path}. Run ff.embed first."
        )
    if not rerun and manifest["analysis"].get("syndication"):
        prev = manifest["analysis"]["syndication"]
        _logger.info("syndication: already computed (%s pairs); pass rerun=True to recompute",
                     prev.get("n_pairs"))
        return prev

    db_path = out / "index.sqlite"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    pair_counts: dict[tuple[str, str], list[float]] = defaultdict(list)
    try:
        # Iterate every claim, query nearest-K via vec0 KNN.
        all_rows = conn.execute(
            "SELECT rowid, item_id, article_id FROM items ORDER BY rowid"
        ).fetchall()
        n = len(all_rows)
        _logger.info("syndication: scanning %d claims, k=%d", n, top_k)

        for i, (rid, _item_id, aid_src) in enumerate(all_rows, start=1):
            v = conn.execute("SELECT embedding FROM vecs WHERE rowid = ?",
                              (rid,)).fetchone()
            if v is None:
                continue
            # sqlite-vec returns L2 distance; for normalized vecs, cos = 1 - d²/2.
            res = conn.execute(
                "SELECT items.article_id, vecs.distance "
                "FROM vecs JOIN items ON items.rowid = vecs.rowid "
                "WHERE vecs.embedding MATCH ? AND vecs.rowid != ? "
                "ORDER BY vecs.distance LIMIT ?",
                (v[0], rid, top_k),
            ).fetchall()
            for aid_neighbor, dist in res:
                if aid_neighbor == aid_src:
                    continue
                cos = 1.0 - (float(dist) ** 2) / 2.0
                if cos < similarity_threshold:
                    continue
                a, b = sorted([aid_src, aid_neighbor])
                pair_counts[(a, b)].append(cos)
            if i % 500 == 0:
                _logger.info("  syndication %d/%d", i, n)
    finally:
        conn.close()

    pairs = []
    for (a, b), sims in pair_counts.items():
        if len(sims) < min_shared_claims:
            continue
        pairs.append({
            "article_a": a,
            "article_b": b,
            "n_shared_claims": len(sims),
            "mean_similarity": sum(sims) / len(sims),
            "max_similarity": max(sims),
        })
    pairs.sort(key=lambda r: (-r["n_shared_claims"], -r["mean_similarity"]))

    syn_path = out / "syndication.jsonl"
    with open(syn_path, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    info = {
        "similarity_threshold": similarity_threshold,
        "min_shared_claims": min_shared_claims,
        "top_k": top_k,
        "n_pairs": len(pairs),
        "output_file": str(syn_path),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest["analysis"]["syndication"] = info
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    _logger.info("syndication done. %d pairs → %s", len(pairs), syn_path)
    return info
