"""Enumerations/constants for regime and method spaces."""

CLASSES = (
    "Balanced",
    "Discovery",
    "Pinning",
    "Shock",
    "Transitional",
)

ORDERS = (
    "Liquidity-Contained",
    "Liquidity-Release",
    "Positioning-Constraint",
    "Information-Override",
    "Correlation-Driven",
)

METHODS = (
    "analytic_local",
    "semi_markov_mixture",
    "quantile_coverage",
    "pde_density",
)


def node_key(cls: str, order: str) -> str:
    """Create a canonical state key."""
    return f"{cls}|{order}"
