"""Magnitude function M_m(t, e, s)."""
from iods.magnitude.magnitude import (
    ContextEmbedding,
    MagnitudeFunction,
    MagnitudeFunctionSoftmax,
    apply_magnitude,
)

__all__ = [
    "ContextEmbedding",
    "MagnitudeFunction",
    "MagnitudeFunctionSoftmax",
    "apply_magnitude",
]
