"""Per-run USD cost cap.

`CostTracker` accumulates per-call LLM costs (thread-safe). When `max_spend`
is hit, `exhausted()` returns True and the worker pool stops submitting new
work. In-flight calls complete, so the total may overshoot by up to
(n_workers - 1) calls. Cache hits and unpriced models contribute $0.
"""

from __future__ import annotations

import threading


class CostTracker:
    """Thread-safe accumulator of per-call LLM costs."""

    __slots__ = ("max_spend", "total", "_lock", "n_calls")

    def __init__(self, max_spend: float | None = None):
        if max_spend is not None and max_spend <= 0:
            raise ValueError(f"max_spend must be > 0; got {max_spend!r}")
        self.max_spend = max_spend
        self.total: float = 0.0
        self.n_calls: int = 0
        self._lock = threading.Lock()

    def add(self, cost_usd: float) -> None:
        """Record a successful call's cost. Negative / non-numeric → no-op."""
        try:
            c = float(cost_usd)
        except (TypeError, ValueError):
            return
        if c < 0:
            return
        with self._lock:
            self.total += c
            self.n_calls += 1

    def exhausted(self) -> bool:
        """True when the cap is set and the total has reached or passed it."""
        if self.max_spend is None:
            return False
        with self._lock:
            return self.total >= self.max_spend

    def snapshot(self) -> dict:
        """Read-only summary for the manifest."""
        with self._lock:
            return {
                "max_spend": self.max_spend,
                "total_usd": round(self.total, 6),
                "n_calls": self.n_calls,
                "exhausted": (
                    self.max_spend is not None and self.total >= self.max_spend
                ),
            }
