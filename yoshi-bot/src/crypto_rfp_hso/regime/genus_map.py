"""Helpers for class/order node key parsing."""

from __future__ import annotations

from crypto_rfp_hso.core.enums import node_key


def split_node_key(key: str) -> tuple[str, str]:
    """Split 'Class|Order' key."""
    if "|" not in key:
        return key, ""
    cls, order = key.split("|", 1)
    return cls, order


def join_node_key(cls: str, order: str) -> str:
    """Join class/order into canonical key."""
    return node_key(cls, order)
