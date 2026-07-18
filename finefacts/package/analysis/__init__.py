"""Analysis subpackage — embedding, clustering, syndication, search.

Optional install: `pip install finefacts[analysis]`
Heavy deps (sentence-transformers, sqlite-vec, hdbscan) are imported lazily
inside each function, so `from finefacts.package.analysis import *` is free
even without the extras installed. The clear ImportError comes when the
function is actually called.

Layout principle: one file per concern, matching DataDreamer's pattern.

Public functions:
    embed(output, ...)              build per-claim embeddings + sqlite-vec index
    cluster(output, ...)            HDBSCAN / k-means over the embeddings
    detect_syndication(output, ...) cross-article high-similarity claim pairs
    search(output, query, ...)      nearest-neighbor lookup at query time

All four operate on a finefacts run directory (one with `manifest.json` from a
prior `ff.extract` run) and extend `manifest.json` append-only with their own
`analysis.<step>` block.
"""

from .cluster import cluster
from .embed import embed
from .search import search
from .syndication import detect_syndication

__all__ = ["embed", "cluster", "detect_syndication", "search"]
