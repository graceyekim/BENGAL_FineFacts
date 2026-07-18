"""Logging setup with API-key redaction.

`get_logger(__name__)` per module; the root `finefacts` logger is
configured once at the start of `ff.extract`. Log records have known
API-key env-var values stripped before write.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path


_SENSITIVE_ENV = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "COHERE_API_KEY", "VOYAGE_API_KEY",
)


class _RedactSecrets(logging.Filter):
    """Strip recognized API keys from log messages.

    A fresh filter instance reads env at construction time. If env values
    change at runtime, call `configure()` again to pick them up.
    """

    def __init__(self):
        super().__init__()
        secrets = [os.environ[k] for k in _SENSITIVE_ENV if os.environ.get(k)]
        self._pattern = (
            re.compile("|".join(re.escape(s) for s in secrets))
            if secrets else None
        )

    def filter(self, record):
        if self._pattern is None:
            return True
        # Apply to formatted message: msg + args
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        redacted = self._pattern.sub("[REDACTED]", msg)
        if redacted != msg:
            record.msg = redacted
            record.args = None
        return True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger factory. Use as `logger = get_logger(__name__)`."""
    return logging.getLogger(name)


def configure(*, run_name: str = "finefacts",
              verbose: bool = False, quiet: bool = False,
              log_file: str | Path | None = None) -> None:
    """Configure the `finefacts` logger hierarchy for one run.

    Safe to call multiple times — existing handlers are removed each call.
    `verbose` raises to DEBUG, `quiet` lowers to WARNING; default is INFO.
    """
    root = logging.getLogger("finefacts")
    for h in list(root.handlers):
        root.removeHandler(h)
    if quiet:
        root.setLevel(logging.WARNING)
    elif verbose:
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)
    root.propagate = False

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(f"[finefacts:{run_name}] %(message)s"))
    ch.addFilter(_RedactSecrets())
    root.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        fh.addFilter(_RedactSecrets())
        root.addHandler(fh)
