"""Duration utilities for dominant-state sequences."""

from __future__ import annotations


def run_length_encode(states: list[str]) -> list[tuple[str, int]]:
    """Run-length encode a state sequence."""
    if not states:
        return []
    out: list[tuple[str, int]] = []
    cur = states[0]
    count = 1
    for s in states[1:]:
        if s == cur:
            count += 1
        else:
            out.append((cur, count))
            cur = s
            count = 1
    out.append((cur, count))
    return out


def durations_by_state(states: list[str]) -> dict[str, list[int]]:
    """Map each state to its observed run lengths."""
    out: dict[str, list[int]] = {}
    for s, d in run_length_encode(states):
        out.setdefault(s, []).append(int(d))
    return out


def exit_pairs(states: list[str]) -> list[tuple[str, str]]:
    """Return (from_state, to_state) pairs for state exits."""
    rle = run_length_encode(states)
    pairs: list[tuple[str, str]] = []
    for i in range(len(rle) - 1):
        s_from, _ = rle[i]
        s_to, _ = rle[i + 1]
        if s_from != s_to:
            pairs.append((s_from, s_to))
    return pairs
