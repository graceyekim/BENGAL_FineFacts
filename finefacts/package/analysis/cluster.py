"""Cluster the vectors built by `ff.embed` — HDBSCAN by default, k-means optional.

Writes the cluster label back to the `items.cluster_id` column of the same
sqlite database, so cluster IDs are joinable to article IDs without re-reading
the per-article JSONs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..log import get_logger

_logger = get_logger(__name__)


_ANALYSIS_EXTRAS_HINT = (
    "ff.cluster requires the [analysis] extras. Run: pip install finefacts[analysis]"
)


def cluster(output, *, algorithm: str = "hdbscan",
            min_cluster_size: int = 10,
            n_clusters: int | None = None,
            rerun: bool = False,
            random_state: int = 42):
    """Cluster the embedded items in a finefacts run dir.

    Args:
        output:           path to a finefacts run dir that has already been
                          embedded (`ff.embed` must have run).
        algorithm:        "hdbscan" (density-based, finds k automatically;
                          default) or "kmeans" (requires n_clusters).
        min_cluster_size: HDBSCAN min cluster size.
        n_clusters:       k-means cluster count; ignored for HDBSCAN.
        rerun:            if False (default) and `analysis.cluster` already in
                          the manifest, skip with a log message.

    Returns the `analysis.cluster` block written to manifest.json.
    """
    try:
        import numpy as np
        import sqlite_vec  # noqa: F401
    except ImportError as e:
        raise ImportError(_ANALYSIS_EXTRAS_HINT) from e

    if algorithm not in ("hdbscan", "kmeans"):
        raise ValueError(f"algorithm must be 'hdbscan' or 'kmeans'; got {algorithm!r}")
    if algorithm == "kmeans" and n_clusters is None:
        raise ValueError("kmeans requires n_clusters")

    out = Path(output).resolve()
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "embed" not in manifest.get("analysis", {}):
        raise RuntimeError(
            f"No embed block in {manifest_path}. Run ff.embed first."
        )
    if not rerun and manifest["analysis"].get("cluster"):
        prev = manifest["analysis"]["cluster"]
        _logger.info("cluster: already clustered (n_clusters=%s); pass rerun=True to recompute",
                     prev.get("n_clusters"))
        return prev

    db_path = out / "index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"No index.sqlite at {db_path}")

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        rows = conn.execute("SELECT rowid, embedding FROM vecs ORDER BY rowid").fetchall()
        if not rows:
            raise RuntimeError(f"No vectors in {db_path}; was ff.embed actually run?")
        rowids = [r[0] for r in rows]
        vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

        _logger.info("cluster: %s over %d vectors", algorithm, len(vecs))
        if algorithm == "hdbscan":
            try:
                import hdbscan
            except ImportError as e:
                raise ImportError(_ANALYSIS_EXTRAS_HINT) from e
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size, metric="euclidean",
            )
            labels = clusterer.fit_predict(vecs)
        else:  # kmeans
            try:
                from sklearn.cluster import KMeans
            except ImportError as e:
                raise ImportError(_ANALYSIS_EXTRAS_HINT) from e
            labels = KMeans(n_clusters=n_clusters, random_state=random_state,
                            n_init="auto").fit_predict(vecs)

        conn.executemany(
            "UPDATE items SET cluster_id = ? WHERE rowid = ?",
            [(int(label), rowid) for rowid, label in zip(rowids, labels)],
        )
        conn.commit()
    finally:
        conn.close()

    unique = set(int(x) for x in labels)
    n_clusters_found = len(unique) - (1 if -1 in unique else 0)
    n_noise = int((labels == -1).sum())

    info = {
        "algorithm": algorithm,
        "min_cluster_size": min_cluster_size if algorithm == "hdbscan" else None,
        "n_clusters": n_clusters_found,
        "n_noise": n_noise,
        "n_vectors": int(len(vecs)),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest["analysis"]["cluster"] = info
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    _logger.info("cluster done. %d clusters, %d noise", n_clusters_found, n_noise)
    return info
