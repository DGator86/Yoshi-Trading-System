"""Node posterior computation and age tracking."""

from __future__ import annotations

from crypto_rfp_hso.core.enums import CLASSES, ORDERS, node_key


def compute_node_posterior(
    w_class: dict[str, float],
    w_order: dict[str, float],
    valid_mask: dict[str, dict[str, int]],
    prev_dom_node: str | None,
    prev_age: int,
) -> tuple[dict[str, float], str, int]:
    """Return (w_node, dom_node, age)."""
    raw: dict[str, float] = {}
    total = 0.0
    for cls in CLASSES:
        pc = float(w_class.get(cls, 0.0))
        for order in ORDERS:
            po = float(w_order.get(order, 0.0))
            valid = int(valid_mask.get(cls, {}).get(order, 1))
            key = node_key(cls, order)
            value = pc * po * float(valid)
            raw[key] = value
            total += value

    if total <= 0.0:
        # Fall back to uniform over valid nodes.
        valid_nodes = [k for k, v in raw.items() if v >= 0.0 and int(valid_mask.get(k.split("|")[0], {}).get(k.split("|")[1], 1)) == 1]
        if not valid_nodes:
            valid_nodes = list(raw.keys())
        uniform = 1.0 / float(len(valid_nodes))
        w_node = {k: (uniform if k in valid_nodes else 0.0) for k in raw.keys()}
    else:
        w_node = {k: float(v / total) for k, v in raw.items()}

    dom_node = max(w_node.items(), key=lambda kv: kv[1])[0]
    if prev_dom_node is not None and dom_node == prev_dom_node:
        age = int(prev_age) + 1
    else:
        age = 1
    return w_node, dom_node, age
