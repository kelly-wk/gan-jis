"""Core JIS projection and audit-safe loss functions.

The default metric normalizes tuple coordinates by the observed range of each
reference table.  This avoids raw millimetre dimensions (for example H and B)
dominating thickness dimensions.  ``normalized=False`` is available only for
legacy comparisons with the recovered experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BLOCK_SIZE = 16
OUTPUT_DIM = 96


def _as_float32_2d(value: np.ndarray, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns or len(array) == 0:
        raise ValueError(f"{name} must have shape (n, {columns}) with n > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _range_scale(table: np.ndarray) -> np.ndarray:
    scale = np.ptp(table, axis=0).astype(np.float32)
    return np.where(scale > 0, scale, 1.0).astype(np.float32)


def _split_numpy(x_mm: np.ndarray) -> tuple[np.ndarray, ...]:
    x = np.asarray(x_mm, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != OUTPUT_DIM:
        raise ValueError(f"x_mm must have shape (n, {OUTPUT_DIM})")
    return tuple(x[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] for i in range(6))


@dataclass(frozen=True)
class JISTables:
    """Allowed beam (H, B, tw, tf) and column (D, t) tuples."""

    beam: np.ndarray
    column: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "beam", _as_float32_2d(self.beam, 4, "beam"))
        object.__setattr__(self, "column", _as_float32_2d(self.column, 2, "column"))

    @classmethod
    def from_csv(cls, beam_csv: str | Path, column_csv: str | Path) -> "JISTables":
        beam = np.genfromtxt(beam_csv, delimiter=",", names=True, dtype=np.float32)
        column = np.genfromtxt(column_csv, delimiter=",", names=True, dtype=np.float32)
        beam_rows = np.column_stack([beam[name] for name in ("H", "B", "tw", "tf")])
        column_rows = np.column_stack([column[name] for name in ("D", "t")])
        return cls(beam_rows, column_rows)

    def _tuples(self, x_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, b, tw, tf, d, t = _split_numpy(x_mm)
        beams = np.stack([h, b, tw, tf], axis=2).reshape(-1, 4)
        columns = np.stack([d, t], axis=2).reshape(-1, 2)
        return beams, columns

    @staticmethod
    def _nearest(
        points: np.ndarray,
        table: np.ndarray,
        *,
        normalized: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        scale = _range_scale(table) if normalized else np.ones(table.shape[1], np.float32)
        delta = (points[:, None, :] - table[None, :, :]) / scale
        squared = np.square(delta).sum(axis=2)
        index = squared.argmin(axis=1)
        return table[index], squared[np.arange(len(points)), index]

    def project(self, x_mm: np.ndarray, *, normalized: bool = True) -> np.ndarray:
        """Project every member tuple to one complete allowed table row."""

        x = np.asarray(x_mm, dtype=np.float32)
        beams, columns = self._tuples(x)
        beam_q, _ = self._nearest(beams, self.beam, normalized=normalized)
        column_q, _ = self._nearest(columns, self.column, normalized=normalized)
        beam_q = beam_q.reshape(len(x), BLOCK_SIZE, 4)
        column_q = column_q.reshape(len(x), BLOCK_SIZE, 2)
        return np.concatenate(
            [
                beam_q[:, :, 0],
                beam_q[:, :, 1],
                beam_q[:, :, 2],
                beam_q[:, :, 3],
                column_q[:, :, 0],
                column_q[:, :, 1],
            ],
            axis=1,
        ).astype(np.float32)

    def projection_distance(
        self,
        x_mm: np.ndarray,
        *,
        normalized: bool = True,
    ) -> np.ndarray:
        """Return one root-mean-square tuple correction per sample."""

        x = np.asarray(x_mm, dtype=np.float32)
        beams, columns = self._tuples(x)
        _, beam_d2 = self._nearest(beams, self.beam, normalized=normalized)
        _, column_d2 = self._nearest(columns, self.column, normalized=normalized)
        per_tuple = np.concatenate(
            [beam_d2.reshape(len(x), BLOCK_SIZE), column_d2.reshape(len(x), BLOCK_SIZE)],
            axis=1,
        )
        return np.sqrt(per_tuple.mean(axis=1)).astype(np.float32)

    def compliance(self, x_mm: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
        """Return whether every beam/column tuple is an allowed complete row."""

        beams, columns = self._tuples(x_mm)
        beam_ok = np.isclose(beams[:, None, :], self.beam[None, :, :], atol=atol).all(axis=2).any(axis=1)
        column_ok = (
            np.isclose(columns[:, None, :], self.column[None, :, :], atol=atol)
            .all(axis=2)
            .any(axis=1)
        )
        n = len(np.asarray(x_mm))
        return np.concatenate(
            [beam_ok.reshape(n, BLOCK_SIZE), column_ok.reshape(n, BLOCK_SIZE)], axis=1
        ).all(axis=1)


def exact_nearest_squared_loss(
    x_mm,
    tables: JISTables,
    *,
    normalized: bool = True,
):
    """Piecewise-differentiable exact nearest-row squared distance.

    Legal rows have exactly zero loss and zero gradient.  This intentionally
    replaces the recovered ``-tau*logsumexp(-d/tau)`` expression, which is not
    a non-negative distance and has a density/entropy bias.
    """

    import torch

    if x_mm.ndim != 2 or x_mm.shape[1] != OUTPUT_DIM:
        raise ValueError(f"x_mm must have shape (n, {OUTPUT_DIM})")
    blocks = [x_mm[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] for i in range(6)]
    beams = torch.stack(blocks[:4], dim=2).reshape(-1, 4)
    columns = torch.stack(blocks[4:], dim=2).reshape(-1, 2)
    beam_table = torch.as_tensor(tables.beam, dtype=x_mm.dtype, device=x_mm.device)
    column_table = torch.as_tensor(tables.column, dtype=x_mm.dtype, device=x_mm.device)
    if normalized:
        beam_scale = torch.as_tensor(_range_scale(tables.beam), dtype=x_mm.dtype, device=x_mm.device)
        column_scale = torch.as_tensor(
            _range_scale(tables.column), dtype=x_mm.dtype, device=x_mm.device
        )
    else:
        beam_scale = torch.ones(4, dtype=x_mm.dtype, device=x_mm.device)
        column_scale = torch.ones(2, dtype=x_mm.dtype, device=x_mm.device)
    beam_d2 = torch.cdist(beams / beam_scale, beam_table / beam_scale).square().min(dim=1).values
    column_d2 = (
        torch.cdist(columns / column_scale, column_table / column_scale)
        .square()
        .min(dim=1)
        .values
    )
    return torch.cat([beam_d2, column_d2]).mean()


def best_of_k_curve(
    tables: JISTables,
    candidates_mm: np.ndarray,
    ks: Iterable[int] = (1, 2, 4, 8, 16),
    *,
    normalized: bool = True,
) -> dict[int, float]:
    """Mean best projection distance for prefixes of a candidate set.

    ``candidates_mm`` has shape ``(conditions, candidates, 96)``.  Selection
    uses projection distance only; it never looks at hidden ground truth.
    """

    candidates = np.asarray(candidates_mm, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[2] != OUTPUT_DIM:
        raise ValueError("candidates_mm must have shape (conditions, candidates, 96)")
    n, k_max, _ = candidates.shape
    distance = tables.projection_distance(
        candidates.reshape(n * k_max, OUTPUT_DIM), normalized=normalized
    ).reshape(n, k_max)
    result: dict[int, float] = {}
    for k in ks:
        if not 1 <= int(k) <= k_max:
            raise ValueError(f"K={k} is outside 1..{k_max}")
        result[int(k)] = float(distance[:, : int(k)].min(axis=1).mean())
    return result


def candidate_selection_audit(
    tables: JISTables,
    candidates_mm: np.ndarray,
    *,
    target_mm: np.ndarray | None = None,
    normalized: bool = True,
) -> dict[str, float | int | str]:
    """Separate deployable selection, projection guarantees, and oracle diagnostics.

    Candidate selection uses only distance to the public reference table.  When a
    hidden target is supplied, the target-based score is returned under an
    explicitly ``oracle`` key and is never used to select the deployable result.
    """

    candidates = np.asarray(candidates_mm, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[2] != OUTPUT_DIM:
        raise ValueError("candidates_mm must have shape (conditions, candidates, 96)")
    n, k, _ = candidates.shape
    flat = candidates.reshape(n * k, OUTPUT_DIM)
    raw_distance = tables.projection_distance(flat, normalized=normalized).reshape(n, k)
    deployable_index = raw_distance.argmin(axis=1)
    selected = candidates[np.arange(n), deployable_index]
    projected = tables.project(selected, normalized=normalized)
    result: dict[str, float | int | str] = {
        "conditions": int(n),
        "candidates_per_condition": int(k),
        "selection_rule": "minimum complete-tuple projection distance; target-independent",
        "raw_pre_projection_combo_compliance_rate_all_candidates": float(
            tables.compliance(flat).mean()
        ),
        "raw_pre_projection_combo_compliance_rate_selected": float(
            tables.compliance(selected).mean()
        ),
        "post_projection_combo_compliance_rate_selected": float(
            tables.compliance(projected).mean()
        ),
        "deployable_selected_projection_distance_mean": float(
            raw_distance[np.arange(n), deployable_index].mean()
        ),
    }
    if target_mm is not None:
        target = np.asarray(target_mm, dtype=np.float32)
        if target.shape != (n, OUTPUT_DIM):
            raise ValueError(f"target_mm must have shape ({n}, {OUTPUT_DIM})")
        l1 = np.abs(candidates - target[:, None, :]).mean(axis=2)
        result.update(
            {
                "oracle_best_l1_diagnostic": float(l1.min(axis=1).mean()),
                "oracle_warning": (
                    "uses hidden ground truth and is unavailable at inference; "
                    "do not report as deployable performance"
                ),
            }
        )
    return result


def illustrative_tables() -> JISTables:
    """Small synthetic tables for tests and demos; these are not official JIS data."""

    return JISTables(
        beam=np.array(
            [
                [200.0, 100.0, 6.0, 9.0],
                [300.0, 150.0, 8.0, 12.0],
                [400.0, 200.0, 10.0, 16.0],
            ],
            dtype=np.float32,
        ),
        column=np.array(
            [[200.0, 6.0], [300.0, 9.0], [400.0, 12.0]], dtype=np.float32
        ),
    )
