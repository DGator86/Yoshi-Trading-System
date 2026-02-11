"""Feature extraction for event-clock buckets."""

from crypto_rfp_hso.features.signature import compute_feature_signature
from crypto_rfp_hso.features.normalize import (
    DEFAULT_SIGNATURE_KEYS,
    signature_to_vector,
)

__all__ = [
    "compute_feature_signature",
    "DEFAULT_SIGNATURE_KEYS",
    "signature_to_vector",
]
