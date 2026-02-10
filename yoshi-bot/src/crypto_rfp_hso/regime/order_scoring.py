"""Order posterior scoring across five constraints."""

from __future__ import annotations

from crypto_rfp_hso.core.math import softmax


def score_orders(sig: dict, temperature: float) -> dict[str, float]:
    """Return w_order over 5 Orders."""
    sigma = abs(float(sig.get("sigma_k", 0.0)))
    mu = abs(float(sig.get("mu_k", 0.0)))
    u2_liq = float(sig.get("U2_liq", 0.5))
    friction = float(sig.get("friction", 0.5))

    u2_perp = float(sig.get("U2_perp", 0.5))
    funding = abs(float(sig.get("funding_k", 0.0)))
    liq_intensity = abs(float(sig.get("liq_intensity_k", 0.0)))
    d_oi = abs(float(sig.get("dOI_k", 0.0)))

    gap_flag = float(sig.get("gap_flag", 0.0))
    liq_cascade = float(sig.get("liq_cascade_flag", 0.0))
    book_vacuum = float(sig.get("book_vacuum", 0.0))

    coupling = float(sig.get("coupling", 0.0))
    corr_to_btc = float(sig.get("corr_to_btc", 0.0))
    idio = float(sig.get("idiosyncratic_strength", (1.0 - coupling) * mu))

    scores = {
        "Liquidity-Contained": +u2_liq - sigma - mu + (1.0 - friction),
        "Liquidity-Release": +(1.0 - u2_liq) + mu + sigma - friction,
        "Positioning-Constraint": +u2_perp + funding + liq_intensity + d_oi,
        "Information-Override": +gap_flag + liq_cascade + book_vacuum,
        "Correlation-Driven": +coupling + corr_to_btc - idio,
    }

    probs = softmax([scores[k] for k in scores.keys()], temperature=temperature)
    return {k: float(v) for k, v in zip(scores.keys(), probs)}
