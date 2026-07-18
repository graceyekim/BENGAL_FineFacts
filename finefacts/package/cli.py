"""`finefacts` CLI — read-only inspection of a run directory.

Subcommands:
    finefacts show <output_dir>             manifest + counts + a sample extraction
    finefacts show <output_dir> --json      same, machine-readable
    finefacts diff <out_a> <out_b>          compare two runs
    finefacts list <parent_dir>             summarize every run under parent_dir
    finefacts list <parent_dir> --json      same, machine-readable

Invoke as `python -m finefacts ...` or via the `finefacts` console script
(declared in pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_manifest(out_dir: Path) -> dict | None:
    mf = out_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _sample_extraction(out_dir: Path) -> tuple[str, dict] | None:
    """Return (article_id, parsed_record) for the first non-error article."""
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "error" in rec or rec.get("_parallel_errors"):
            continue
        return (rec.get("article_id", p.stem), rec)
    return None


def _print_kv(rows, indent=2):
    pad = max(len(k) for k, _ in rows) if rows else 0
    for k, v in rows:
        print(f"{' ' * indent}{k:<{pad}}  {v}")


def cmd_show(args):
    out = Path(args.output_dir).resolve()
    if not out.is_dir():
        print(f"error: {out} is not a directory", file=sys.stderr)
        return 2
    manifest = _load_manifest(out)
    if manifest is None:
        print(f"error: no manifest.json at {out}", file=sys.stderr)
        return 2

    if getattr(args, "json", False):
        sample = _sample_extraction(out)
        payload = {
            "manifest": manifest,
            "sample_extraction": (
                {"article_id": sample[0], "record": sample[1]}
                if sample else None
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"\n=== finefacts run @ {out} ===\n")
    print("Manifest")
    rows = [
        ("name",            manifest.get("name", "?")),
        ("library_version", manifest.get("library_version", "?")),
        ("model",           manifest.get("model", "?")),
        ("mode",            manifest.get("mode", "?")),
        ("n_prompts",       manifest.get("n_prompts", "?")),
        ("n_workers",       manifest.get("n_workers", "?")),
        ("prompt_hash",     manifest.get("prompt_hash", "?")),
        ("started_at",      manifest.get("started_at", "?")),
        ("finished_at",     manifest.get("finished_at", "?")),
        ("n_total",         manifest.get("n_articles_total", "?")),
        ("n_succeeded",     manifest.get("n_articles_succeeded", "?")),
        ("n_failed",        manifest.get("n_articles_failed", "?")),
    ]
    if manifest.get("schema"):
        rows.append(("schema", manifest["schema"].get("class_name", "?")))
    if manifest.get("max_input_tokens") is not None:
        rows.append(("max_input_tokens", manifest["max_input_tokens"]))
    _print_kv(rows)

    if manifest.get("analysis"):
        print("\nAnalysis steps")
        for step, info in manifest["analysis"].items():
            print(f"  {step}:")
            _print_kv(list(info.items()), indent=4)

    if manifest.get("mode") == "parallel" and manifest.get("parallel_keys"):
        print(f"\nParallel keys: {', '.join(manifest['parallel_keys'])}")

    sample = _sample_extraction(out)
    if sample:
        aid, rec = sample
        print(f"\nSample extraction ({aid})")
        ext_preview = json.dumps(rec.get("extracted"), ensure_ascii=False, indent=2)
        if len(ext_preview) > 800:
            ext_preview = ext_preview[:800] + "  ... [truncated]"
        for line in ext_preview.splitlines():
            print(f"  {line}")
    print()
    return 0


def cmd_diff(args):
    a = Path(args.output_dir_a).resolve()
    b = Path(args.output_dir_b).resolve()
    ma = _load_manifest(a)
    mb = _load_manifest(b)
    if ma is None or mb is None:
        print("error: one or both run dirs lack manifest.json", file=sys.stderr)
        return 2

    print(f"\n=== diff {a} ↔ {b} ===\n")
    keys = ("name", "model", "mode", "n_prompts", "prompt_hash",
            "n_articles_total", "n_articles_succeeded", "n_articles_failed")
    rows = []
    pad = max(len(k) for k in keys)
    print(f"{'':<{pad+2}}  {'A':<30}  {'B':<30}")
    print(f"{'':<{pad+2}}  {'-' * 30}  {'-' * 30}")
    for k in keys:
        va, vb = ma.get(k, "?"), mb.get(k, "?")
        marker = "  " if va == vb else "* "
        print(f"{marker}{k:<{pad}}  {str(va):<30}  {str(vb):<30}")
    print()
    return 0


def cmd_list(args):
    from .manifest import list_runs
    parent = Path(args.parent_dir).resolve()
    if not parent.is_dir():
        print(f"error: {parent} is not a directory", file=sys.stderr)
        return 2
    runs = list_runs(parent)

    if getattr(args, "json", False):
        print(json.dumps(runs, indent=2, ensure_ascii=False))
        return 0

    if not runs:
        print(f"No finefacts runs found under {parent}.")
        return 0

    print(f"\n=== finefacts runs under {parent} ({len(runs)} found) ===\n")
    print(f"{'NAME':<30}  {'MODEL':<28}  {'MODE':<10}  {'OK/TOT':<8}  {'COST':<10}  FINISHED")
    print(f"{'-' * 30}  {'-' * 28}  {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 19}")
    for r in runs:
        ok = f"{r['n_succeeded']}/{r['n_succeeded'] + r['n_failed']}"
        cost = (
            f"${r['cost_usd']:.4f}" if r['cost_usd'] is not None else "—"
        )
        finished = r['finished_at'][:19] if r['finished_at'] else "—"
        print(f"{r['name']:<30.30}  {r['model']:<28.28}  "
              f"{r['mode']:<10.10}  {ok:<8}  {cost:<10}  {finished}")
    print()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="finefacts",
                                 description="Inspect finefacts run directories.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("show", help="Show manifest + a sample extraction.")
    ps.add_argument("output_dir", help="Path to a finefacts output dir.")
    ps.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON instead of formatted text.")
    ps.set_defaults(func=cmd_show)

    pd = sub.add_parser("diff", help="Compare two run directories.")
    pd.add_argument("output_dir_a")
    pd.add_argument("output_dir_b")
    pd.set_defaults(func=cmd_diff)

    pl = sub.add_parser("list", help="Summarize every finefacts run under a parent dir.")
    pl.add_argument("parent_dir", help="Path containing one or more run dirs.")
    pl.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON instead of formatted text.")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
