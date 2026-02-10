"""Realtime single-step pipeline update."""

from __future__ import annotations

from crypto_rfp_hso.pipelines.build_history import build_history


def realtime_step(
    state: dict,
    new_bucket: dict,
    new_l2_snapshot: dict | None,
    new_perp_metrics: dict | None,
    coupling_inputs: dict | None = None,
) -> dict:
    """Append latest data and rebuild deterministic projections."""
    buckets = list(state.get("buckets", []))
    l2_snapshots = list(state.get("l2_snapshots", []))
    perp_metrics = list(state.get("perp_metrics", []))
    config = dict(state.get("config", {}))
    templates = state.get("templates")
    valid_mask = state.get("valid_mask")

    buckets.append(new_bucket)
    if new_l2_snapshot is not None:
        l2_snapshots.append(new_l2_snapshot)
    if new_perp_metrics is not None:
        perp_metrics.append(new_perp_metrics)

    built = build_history(
        buckets=buckets,
        l2_snapshots=l2_snapshots,
        perp_metrics=perp_metrics,
        coupling_inputs=coupling_inputs or {},
        config=config,
        templates=templates,
        valid_mask=valid_mask,
    )
    built["buckets"] = buckets
    built["l2_snapshots"] = l2_snapshots
    built["perp_metrics"] = perp_metrics
    built["valid_mask"] = valid_mask
    return built
