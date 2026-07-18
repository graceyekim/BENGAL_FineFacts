"""Per-run manifest (`manifest.json`) and the loader / diff / export helpers.

Every output directory carries a manifest with the prompt fingerprint,
model, library version, timings, and success/fail counts.
`manifest_version = 1` today; readers can dispatch on it as it evolves.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


MANIFEST_VERSION = 1


def build_manifest(name, model, prompt, max_article_chars,
                   distill, distill_model, gold_size):
    """Skeleton manifest written at the start of a run."""
    from .. import __version__  # function-local to avoid an import cycle
    return {
        "manifest_version": MANIFEST_VERSION,
        "name": name,
        "library_version": __version__,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "prompt_preview": prompt[:200],
        "max_article_chars": max_article_chars,
        "distill": distill,
        "distill_model": distill_model if distill else None,
        "gold_size": gold_size if distill else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def retry_failed(output_dir) -> dict:
    """Delete per-article JSONs that have errors so a re-run of `ff.extract` on
    the same corpus + output dir re-attempts only those records.

    A record is failed if it has an `error` key (chained mode) or any value in
    `_parallel_errors` (parallel mode). The corresponding `.json` file is
    deleted; the cache (if present) is preserved — successful sub-calls in
    parallel records remain usable on the re-attempt.

    Returns:
        {"n_deleted": int, "deleted_ids": list[str]}
    """
    p = Path(output_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")
    deleted = []
    for f in sorted(p.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Treat unreadable files as failed too — delete so they're re-attempted.
            deleted.append(f.stem)
            f.unlink()
            continue
        if "error" in rec or rec.get("_parallel_errors"):
            deleted.append(rec.get("article_id", f.stem))
            f.unlink()
    return {"n_deleted": len(deleted), "deleted_ids": deleted}


def load(output_dir, *, include_errors: bool = True):
    """Load every per-article record from a finefacts run dir into a list.

    Saves users the boilerplate:
        [json.loads(f.read_text()) for f in Path(out).glob("*.json") if f.name != "manifest.json"]

    Args:
        output_dir:     path to a finefacts output directory.
        include_errors: if False, records with `error` or `_parallel_errors`
                        are filtered out. Default True (return everything).

    Returns:
        list of parsed record dicts, ordered by article_id.
    """
    p = Path(output_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")
    out = []
    for f in sorted(p.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not include_errors and ("error" in rec or rec.get("_parallel_errors")):
            continue
        out.append(rec)
    return out


def _get_by_path(rec, dotted_path):
    """Walk `dotted_path` through nested dicts; return None on miss."""
    v = rec
    for p in dotted_path.split("."):
        if not isinstance(v, dict):
            return None
        v = v.get(p)
    return v


def _stringify(v):
    """Return a searchable string for a leaf value — str verbatim, else JSON."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    return json.dumps(v, ensure_ascii=False)


def match_by_field(field: str):
    """Matcher: two extractions agree iff their value at `field` is equal.

    `field` may be dotted (e.g. `"extracted.topic"`).
    """
    def check(a_ext, b_ext):
        return _get_by_path(a_ext, field) == _get_by_path(b_ext, field)
    check.__name__ = f"match_by_field({field!r})"
    return check


def match_by_keywords(keywords, *, field=None, mode: str = "any"):
    """Matcher: agree iff both records mention (or both don't) any / all keywords.

    `field` is a dotted path to restrict the search; None searches the
    whole extraction.
    """
    kws = [str(k).lower() for k in keywords]
    if not kws:
        raise ValueError("match_by_keywords requires at least one keyword")
    if mode not in ("any", "all"):
        raise ValueError(f"mode must be 'any' or 'all'; got {mode!r}")

    def _hit(ext):
        text = _stringify(_get_by_path(ext, field) if field else ext).lower()
        if mode == "all":
            return all(k in text for k in kws)
        return any(k in text for k in kws)

    def check(a_ext, b_ext):
        return _hit(a_ext) == _hit(b_ext)
    check.__name__ = f"match_by_keywords({kws[:3]!r}{', ...' if len(kws) > 3 else ''})"
    return check


def match_by_regex(pattern, *, field=None):
    """Matcher: agree iff both records match `pattern` (or neither does).

    Matched with re.IGNORECASE | re.MULTILINE. `field` is a dotted path.
    """
    r = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def check(a_ext, b_ext):
        a_txt = _stringify(_get_by_path(a_ext, field) if field else a_ext)
        b_txt = _stringify(_get_by_path(b_ext, field) if field else b_ext)
        return bool(r.search(a_txt)) == bool(r.search(b_txt))
    check.__name__ = f"match_by_regex({pattern!r})"
    return check


def compare_runs(dir_a, dir_b, *, match=None) -> dict:
    """Per-article diff between two run directories.

    For each article_id present in either run, categorize it into one of
    four buckets. Useful for A/B prompt or model testing where you ran the
    same corpus twice with a config tweak and want to see what changed.

    Returns:
        {
            "only_in_a":     [article_id, ...],
            "only_in_b":     [article_id, ...],
            "identical":     [article_id, ...],
            "differing":     [{"article_id", "a_extracted", "b_extracted"}, ...],
            "counts":        {"only_a", "only_b", "identical", "differing", "total"},
        }
    """
    a_recs = {r.get("article_id"): r for r in load(dir_a)}
    b_recs = {r.get("article_id"): r for r in load(dir_b)}
    a_ids, b_ids = set(a_recs), set(b_recs)

    only_a = sorted(a_ids - b_ids)
    only_b = sorted(b_ids - a_ids)
    common = sorted(a_ids & b_ids)

    # If no custom matcher provided, use strict dict equality (default).
    def _strict_match(a, b):
        return a == b
    matcher = match if match is not None else _strict_match

    identical = []
    differing = []
    for aid in common:
        # Compare only the extracted content, ignore per-record metadata
        # (extraction_date, model, timestamps) which vary trivially.
        a_ext = a_recs[aid].get("extracted")
        b_ext = b_recs[aid].get("extracted")
        try:
            same = bool(matcher(a_ext, b_ext))
        except Exception:
            # Bad matcher → treat as differing (safer than crashing the report)
            same = False
        if same:
            identical.append(aid)
        else:
            differing.append({
                "article_id": aid,
                "a_extracted": a_ext,
                "b_extracted": b_ext,
            })

    # Summary rates — useful for "did the candidate match the reference?"
    # (e.g. base local model vs Claude gold — decides whether distillation
    # is worth the training cost).
    n_common = len(common)
    exact_match_rate = (len(identical) / n_common) if n_common else 0.0
    coverage_a_to_b = (n_common / len(a_ids)) if a_ids else 1.0
    coverage_b_to_a = (n_common / len(b_ids)) if b_ids else 1.0

    return {
        "only_in_a": only_a,
        "only_in_b": only_b,
        "identical": identical,
        "differing": differing,
        "counts": {
            "only_a": len(only_a),
            "only_b": len(only_b),
            "identical": len(identical),
            "differing": len(differing),
            "total": len(a_ids | b_ids),
        },
        "rates": {
            # Of articles present in both runs, what fraction produced
            # identical `extracted` content. 1.0 = candidate matches
            # reference perfectly; 0.0 = every extraction differs.
            "exact_match_rate": exact_match_rate,
            # What fraction of A's articles are also in B (and vice versa).
            "coverage_a_in_b": coverage_a_to_b,
            "coverage_b_in_a": coverage_b_to_a,
        },
    }


def _flatten(rec: dict, prefix: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict for CSV export. Lists get JSON-encoded as one cell."""
    flat = {}
    for k, v in rec.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key, sep))
        elif isinstance(v, list):
            flat[key] = json.dumps(v, ensure_ascii=False)
        else:
            flat[key] = v
    return flat


def to_jsonl(output_dir, out_path=None, *, include_errors: bool = True) -> Path:
    """Emit all per-article records to a single JSONL file. Preserves nested structure.

    Args:
        output_dir:     the finefacts run dir to read from.
        out_path:       destination file. Defaults to `<output_dir>.jsonl`
                        alongside the run dir.
        include_errors: forwarded to `load()`.

    Returns the written path.
    """
    p = Path(output_dir)
    out_path = Path(out_path) if out_path else p.with_suffix(".jsonl")
    recs = load(p, include_errors=include_errors)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def to_csv(output_dir, out_path=None, *, include_errors: bool = True) -> Path:
    """Emit all per-article records to a flat CSV.

    Nested dicts flatten via dot-notation (`extracted.claims` -> column).
    Lists are JSON-encoded into one cell — CSV can't represent them natively.

    Args:
        output_dir:     the finefacts run dir.
        out_path:       destination file. Defaults to `<output_dir>.csv`.
        include_errors: forwarded to `load()`.

    Returns the written path.
    """
    import csv as _csv
    p = Path(output_dir)
    out_path = Path(out_path) if out_path else p.with_suffix(".csv")
    recs = [_flatten(r) for r in load(p, include_errors=include_errors)]
    if not recs:
        out_path.write_text("", encoding="utf-8")
        return out_path
    # Union of columns across all records; preserve first-seen order for reader stability.
    columns = []
    seen = set()
    for r in recs:
        for k in r:
            if k not in seen:
                columns.append(k); seen.add(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    return out_path


def show(output_dir):
    """Return a Python dict summarizing a finefacts run dir.

    Same data the `finefacts show <out>` CLI prints, available programmatically
    for use in notebooks / scripts.

    Returns:
        {
            "manifest": <full manifest dict>,
            "sample_extraction": {"article_id": str, "record": dict} | None,
        }
    """
    p = Path(output_dir)
    mf = p / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"No manifest.json at {mf}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    # First non-error article
    sample = None
    for f in sorted(p.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "error" in rec or rec.get("_parallel_errors"):
            continue
        sample = {"article_id": rec.get("article_id", f.stem), "record": rec}
        break
    return {"manifest": manifest, "sample_extraction": sample}


def list_runs(parent_dir):
    """Scan `parent_dir` for finefacts run directories and return a summary list.

    A "run dir" is any subdirectory containing a `manifest.json`. Returns a
    list of dicts with the manifest's key fields, ordered by `finished_at`
    descending. Subdirectories without a manifest are skipped silently.

    Args:
        parent_dir: directory containing one or more run subdirectories.

    Returns:
        list of {name, path, model, mode, n_succeeded, n_failed,
                 started_at, finished_at, cost_usd}
    """
    p = Path(parent_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")
    out = []
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            continue
        mf = child / "manifest.json"
        if not mf.exists():
            continue
        try:
            man = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append({
            "name": man.get("name", child.name),
            "path": str(child),
            "model": man.get("model", "?"),
            "mode": man.get("mode", "?"),
            "n_succeeded": man.get("n_articles_succeeded", 0),
            "n_failed": man.get("n_articles_failed", 0),
            "started_at": man.get("started_at", ""),
            "finished_at": man.get("finished_at", ""),
            "cost_usd": (
                man.get("cost_tracking", {}).get("total_usd")
                if isinstance(man.get("cost_tracking"), dict) else None
            ),
        })
    out.sort(key=lambda r: r["finished_at"], reverse=True)
    return out


def finalize_manifest(out: Path, manifest: dict) -> None:
    """Count success/fail per-article JSONs in `out` and write `manifest.json`.

    A record counts as failed if it has a chained "error" key OR any
    "_parallel_errors" (parallel-mode failure for at least one key).
    """
    n_total = n_failed = 0
    for p in out.glob("*.json"):
        if p.name == "manifest.json":
            continue
        n_total += 1
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if "error" in rec or rec.get("_parallel_errors"):
                n_failed += 1
        except (json.JSONDecodeError, OSError):
            n_failed += 1
    manifest.update({
        "n_articles_total": n_total,
        "n_articles_failed": n_failed,
        "n_articles_succeeded": n_total - n_failed,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    })
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
