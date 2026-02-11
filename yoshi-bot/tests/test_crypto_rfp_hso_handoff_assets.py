"""Keep handoff artifacts synchronized with code constants."""

import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_rfp_hso.core.event_time import EVENT_ALPHABET  # noqa: E402
from crypto_rfp_hso.core.schemas import DEFAULT_CONFIG  # noqa: E402
from crypto_rfp_hso.projection.aggregator import G, ORDER_METHOD_MULT  # noqa: E402
from crypto_rfp_hso.regime.validity_mask import VALID_MASK  # noqa: E402


def _nested_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_nested_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_nested_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def test_crypto_rfp_hso_handoff_yaml_matches_code_constants():
    path = Path(__file__).parent.parent / "configs" / "crypto_rfp_hso.yaml"
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    cfg = payload["crypto_rfp_hso"]
    assert _nested_equal(cfg["defaults"], DEFAULT_CONFIG)
    assert _nested_equal(cfg["valid_mask"], VALID_MASK)
    assert _nested_equal(cfg["method_gating"]["G"], G)
    assert _nested_equal(cfg["method_gating"]["order_method_mult"], ORDER_METHOD_MULT)
    assert cfg["event_alphabet"] == list(EVENT_ALPHABET)
