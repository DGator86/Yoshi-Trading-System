"""Validity mask for class x order nodes."""

VALID_MASK = {
    "Balanced": {
        "Liquidity-Contained": 1,
        "Liquidity-Release": 1,
        "Positioning-Constraint": 1,
        "Information-Override": 1,
        "Correlation-Driven": 1,
    },
    "Discovery": {
        "Liquidity-Contained": 1,
        "Liquidity-Release": 1,
        "Positioning-Constraint": 1,
        "Information-Override": 1,
        "Correlation-Driven": 1,
    },
    "Pinning": {
        "Liquidity-Contained": 1,
        "Liquidity-Release": 1,
        "Positioning-Constraint": 1,
        "Information-Override": 1,
        "Correlation-Driven": 1,
    },
    "Shock": {
        "Liquidity-Contained": 0,
        "Liquidity-Release": 1,
        "Positioning-Constraint": 1,
        "Information-Override": 1,
        "Correlation-Driven": 1,
    },
    "Transitional": {
        "Liquidity-Contained": 1,
        "Liquidity-Release": 1,
        "Positioning-Constraint": 1,
        "Information-Override": 0,
        "Correlation-Driven": 1,
    },
}
