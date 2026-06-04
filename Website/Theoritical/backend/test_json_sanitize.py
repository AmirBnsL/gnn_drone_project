"""Tests for JSON sanitization of non-finite floats."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from json_sanitize import finite_or_none, sanitize_for_json


def test_sanitize_inf_to_none():
    out = sanitize_for_json({"a": float("inf"), "b": [1.0, float("nan")]})
    assert out["a"] is None
    assert out["b"][0] == 1.0
    assert out["b"][1] is None
    json.dumps(out)


def test_finite_or_none():
    assert finite_or_none(1.5) == 1.5
    assert finite_or_none(float("inf")) is None


def _run_all() -> None:
    test_sanitize_inf_to_none()
    test_finite_or_none()
    print("test_json_sanitize: all passed")


if __name__ == "__main__":
    _run_all()
