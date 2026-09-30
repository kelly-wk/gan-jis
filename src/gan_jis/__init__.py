"""Audited GAN-JIS reproduction utilities."""

from .core import (
    JISTables,
    best_of_k_curve,
    candidate_selection_audit,
    exact_nearest_squared_loss,
    illustrative_tables,
)

__all__ = [
    "JISTables",
    "best_of_k_curve",
    "candidate_selection_audit",
    "exact_nearest_squared_loss",
    "illustrative_tables",
]

__version__ = "0.1.0"
