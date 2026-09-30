from __future__ import annotations

import unittest

import numpy as np

from gan_jis import (
    JISTables,
    best_of_k_curve,
    candidate_selection_audit,
    exact_nearest_squared_loss,
    illustrative_tables,
)


def legal_vector(tables: JISTables, beam_index: int = 0, column_index: int = 0) -> np.ndarray:
    beam = tables.beam[beam_index]
    column = tables.column[column_index]
    return np.concatenate(
        [np.repeat(beam[i], 16) for i in range(4)]
        + [np.repeat(column[i], 16) for i in range(2)]
    ).astype(np.float32)


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = illustrative_tables()

    def test_projection_is_compliant_and_idempotent(self) -> None:
        x = legal_vector(self.tables)[None, :] + 3.0
        projected = self.tables.project(x)
        self.assertTrue(self.tables.compliance(projected).all())
        np.testing.assert_array_equal(projected, self.tables.project(projected))

    def test_best_of_k_is_monotone_nonincreasing(self) -> None:
        exact = legal_vector(self.tables)
        candidates = np.stack([exact + delta for delta in (20.0, 10.0, 3.0, 0.0)], axis=0)
        candidates = np.stack([candidates, candidates + 1.0], axis=0)
        curve = best_of_k_curve(self.tables, candidates, ks=(1, 2, 4))
        self.assertGreaterEqual(curve[1], curve[2])
        self.assertGreaterEqual(curve[2], curve[4])

    def test_selection_audit_separates_raw_projected_and_oracle_metrics(self) -> None:
        exact = legal_vector(self.tables)
        candidates = np.stack([exact + 20.0, exact + 2.0], axis=0)[None, :, :]
        report = candidate_selection_audit(
            self.tables,
            candidates,
            target_mm=(exact + 18.0)[None, :],
        )
        self.assertEqual(report["raw_pre_projection_combo_compliance_rate_all_candidates"], 0.0)
        self.assertEqual(report["post_projection_combo_compliance_rate_selected"], 1.0)
        self.assertGreater(report["oracle_best_l1_diagnostic"], 0.0)
        self.assertIn("target-independent", report["selection_rule"])

    def test_legal_rows_have_zero_loss_and_zero_gradient(self) -> None:
        import torch

        x = torch.tensor(legal_vector(self.tables)[None, :], requires_grad=True)
        loss = exact_nearest_squared_loss(x, self.tables)
        loss.backward()
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(x.grad.abs().max().item()), 0.0)

    def test_perturbation_has_positive_loss_and_descent_points_to_nearest_row(self) -> None:
        import torch

        base = legal_vector(self.tables)
        base[0] += 5.0
        x = torch.tensor(base[None, :], requires_grad=True)
        loss = exact_nearest_squared_loss(x, self.tables)
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(x.grad[0, 0].item()), 0.0)
        step = x.grad.detach() / x.grad.detach().abs().max().clamp_min(1e-12)
        moved = x.detach() - step
        self.assertLess(
            float(exact_nearest_squared_loss(moved, self.tables).item()),
            float(loss.item()),
        )


if __name__ == "__main__":
    unittest.main()
