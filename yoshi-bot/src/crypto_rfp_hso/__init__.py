"""Crypto Regime Field Probability Hilbert Space Overlay package.

This package implements a print-first, event-clock crypto prediction engine
with Hilbert regime projection, semi-Markov walkforward propagation, and
forward collapse fan generation.
"""

from crypto_rfp_hso.data.bucketize import bucketize_prints_notional
from crypto_rfp_hso.features.signature import compute_feature_signature
from crypto_rfp_hso.hilbert.project import hilbert_project
from crypto_rfp_hso.regime.order_scoring import score_orders
from crypto_rfp_hso.regime.node_posterior import compute_node_posterior
from crypto_rfp_hso.fractal.hurst_ghe import hurst_ghe
from crypto_rfp_hso.transitions.semimarkov_fit import fit_semi_markov_params
from crypto_rfp_hso.transitions.semimarkov_propagate import propagate_semi_markov
from crypto_rfp_hso.projection.per_state_models import fit_per_state_t_params
from crypto_rfp_hso.projection.fan import build_forward_fan
from crypto_rfp_hso.overlay.payload import build_overlay_payload
from crypto_rfp_hso.core.event_time import EVENT_ALPHABET, build_event_time_index
from crypto_rfp_hso.dynamics.discrete import map_path_and_bands
from crypto_rfp_hso.dynamics.consensus import venue_consensus_update
from crypto_rfp_hso.projection.hazard_surface import hazard_surface_distribution

__all__ = [
    "bucketize_prints_notional",
    "compute_feature_signature",
    "hilbert_project",
    "score_orders",
    "compute_node_posterior",
    "hurst_ghe",
    "fit_semi_markov_params",
    "propagate_semi_markov",
    "fit_per_state_t_params",
    "build_forward_fan",
    "build_overlay_payload",
    "EVENT_ALPHABET",
    "build_event_time_index",
    "map_path_and_bands",
    "venue_consensus_update",
    "hazard_surface_distribution",
]
