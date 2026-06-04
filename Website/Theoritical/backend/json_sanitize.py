"""JSON-safe serialization helpers (replace non-finite floats)."""

from __future__ import annotations

import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace inf/nan floats with None for JSON compliance."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def finite_or_none(value: float) -> float | None:
    """Return value if finite, else None (for telemetry fields)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
