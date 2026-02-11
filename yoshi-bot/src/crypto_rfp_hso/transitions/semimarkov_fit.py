"""Semi-Markov parameter fitting."""

from __future__ import annotations

import math

from crypto_rfp_hso.transitions.durations import durations_by_state, exit_pairs


def _fit_powerlaw_alpha(durations: list[int], tau_min: int) -> float:
    filtered = [float(d) for d in durations if d >= tau_min]
    n = len(filtered)
    if n < 2:
        return 2.0
    denom = 0.0
    t0 = float(max(tau_min, 1))
    for d in filtered:
        denom += math.log(max(d / t0, 1.0))
    if denom <= 0.0:
        return 2.0
    return 1.0 + n / denom


def fit_semi_markov_params(
    dom_nodes: list[str],
    all_states: list[str],
    tau_min: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (alpha_by_state, T_exit)."""
    states = list(all_states)
    if not states:
        return {}, {}

    durations = durations_by_state(dom_nodes)
    alpha_by_state = {s: float(_fit_powerlaw_alpha(durations.get(s, []), tau_min=tau_min)) for s in states}

    # Exit transition counts (diagonal forced to 0).
    counts: dict[str, dict[str, float]] = {s: {j: 0.0 for j in states} for s in states}
    for s_from, s_to in exit_pairs(dom_nodes):
        if s_from in counts and s_to in counts[s_from] and s_from != s_to:
            counts[s_from][s_to] += 1.0

    t_exit: dict[str, dict[str, float]] = {}
    for s in states:
        row = counts[s]
        row[s] = 0.0
        total = sum(v for j, v in row.items() if j != s)
        if total <= 0.0:
            # Uniform over other states.
            m = max(len(states) - 1, 1)
            t_exit[s] = {j: (0.0 if j == s else 1.0 / m) for j in states}
        else:
            t_exit[s] = {j: (0.0 if j == s else row[j] / total) for j in states}
    return alpha_by_state, t_exit
