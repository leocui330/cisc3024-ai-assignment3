"""Check the paper update, pixel graph and reproducibility boundaries."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import contaminate, partitions, predict
from reducers import grid_laplacian, mm_direction, principal_axes, robust_axes


class ReducerTests(unittest.TestCase):
    def test_grid_penalty_is_neighbor_difference_sum(self):
        lap = grid_laplacian(2, 3)
        a = np.array([[0., 1., 2.], [3., 4., 5.]])
        expected = np.square(np.diff(a, axis=0)).sum() + np.square(np.diff(a, axis=1)).sum()
        np.testing.assert_array_equal(lap @ np.ones(6), np.zeros(6))
        self.assertAlmostEqual(a.ravel() @ lap @ a.ravel(), expected)

    def test_symmetric_update_matches_paper_equation_with_zero_loadings(self):
        lap = grid_laplacian(2, 2)
        w = np.array([-0.6, 0., 0.8, 0.])
        v = np.array([3., -2., 7., 1.])
        u = np.diag(np.abs(w))
        eta1, eta2 = .1, .7
        expected = np.linalg.solve(u + eta1 * np.eye(4) + eta2 * (u @ lap), u @ v)
        np.testing.assert_allclose(mm_direction(w, v, lap, eta1, eta2), expected, atol=1e-13)
        np.testing.assert_array_equal(mm_direction(w, v, lap, 0., 0.), v)
        np.testing.assert_allclose(mm_direction(w, v, lap, 0., eta2),
                                   np.linalg.solve(np.eye(4) + eta2 * lap, v))

    def test_rank1_optimum_and_predictions(self):
        x = np.array([[-3., 0., 0., 0.], [-1., 0., 0., 0.],
                      [1., 0., 0., 0.], [3., 0., 0., 0.]])
        initial = principal_axes(x, 1)
        w, i, change = robust_axes(x, initial, grid_laplacian(2, 2), 0., 0.)
        np.testing.assert_allclose(w[:, 0], [1., 0., 0., 0.], atol=1e-13)
        self.assertEqual(i, [1])
        self.assertEqual(change, [0.0])
        np.testing.assert_array_equal(predict(x, x, np.arange(4), w), np.arange(4))

    def test_split_and_corruption_are_reproducible(self):
        y = np.resize(np.arange(10), 3823)
        fit, val = partitions(y, 22)
        self.assertEqual((len(fit), len(val)), (3058, 765))
        self.assertEqual(np.intersect1d(fit, val).size, 0)
        x = np.full((3058, 64), .5)
        a, damage = contaminate(x, 11)
        b, _ = contaminate(x, 11)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(damage["cells_changed"], 917 * 22)
        self.assertTrue(np.all(x == .5))


if __name__ == "__main__":
    unittest.main()
