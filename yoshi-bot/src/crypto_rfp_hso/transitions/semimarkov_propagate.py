"""Semi-Markov walkforward propagation with duration hazard."""

from __future__ import annotations

from crypto_rfp_hso.core.math import clamp


def _normalize(pi: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in pi.values())
    if total <= 0.0:
        n = max(len(pi), 1)
        return {k: 1.0 / n for k in pi.keys()}
    return {k: float(v) / total for k, v in pi.items()}


def _hazard(alpha: float, age: int, h_min: float, h_max: float) -> float:
    a = max(int(age), 1)
    al = max(float(alpha), 1e-6)
    h = 1.0 - ((a / (a + 1.0)) ** al)
    return clamp(h, h_min, h_max)


def propagate_semi_markov(
    pi0: dict[str, float],
    dom_state: str,
    age0: int,
    alpha_by_state: dict[str, float],
    T_exit: dict[str, dict[str, float]],
    horizons: list[int],
) -> dict[int, dict[str, float]]:
    """Return pi_forward[tau][state]."""
    if not pi0:
        return {}

    states = list(pi0.keys())
    max_h = max([int(h) for h in horizons] or [1])
    wanted = set(int(h) for h in horizons)

    h_min = 0.01
    h_max = 0.99

    pi = _normalize({s: float(pi0.get(s, 0.0)) for s in states})
    current_dom = dom_state if dom_state in pi else max(pi.items(), key=lambda kv: kv[1])[0]
    current_age = max(int(age0), 1)

    out: dict[int, dict[str, float]] = {}

    for step in range(1, max_h + 1):
        nxt = {s: 0.0 for s in states}
        for s in states:
            alpha = float(alpha_by_state.get(s, 2.0))
            age = current_age if s == current_dom else 1
            hazard = _hazard(alpha, age, h_min=h_min, h_max=h_max)
            stay = 1.0 - hazard
            nxt[s] += pi[s] * stay

            row = T_exit.get(s, {})
            row_total = sum(float(row.get(j, 0.0)) for j in states if j != s)
            if row_total <= 0.0:
                m = max(len(states) - 1, 1)
                for j in states:
                    if j == s:
                        continue
                    nxt[j] += pi[s] * hazard * (1.0 / m)
            else:
                for j in states:
                    if j == s:
                        continue
                    nxt[j] += pi[s] * hazard * (float(row.get(j, 0.0)) / row_total)

        pi = _normalize(nxt)
        new_dom = max(pi.items(), key=lambda kv: kv[1])[0]
        if new_dom == current_dom:
            current_age += 1
        else:
            current_dom = new_dom
            current_age = 1

        if step in wanted:
            out[step] = dict(pi)

    return out
