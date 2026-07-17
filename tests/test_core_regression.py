import unittest

import numpy as np

from ect_analysis.channels import greedy_regularized_logdet
from ect_analysis.mesh import make_common_grid
from ect_analysis.metrics import (
    gram_and_eigenvalues,
    regularized_spectral_metrics,
    weighted_gini,
)
from ect_analysis.models import MeshMeta


class CoreRegressionTest(unittest.TestCase):
    def test_frozen_numerical_signature(self) -> None:
        rng = np.random.default_rng(20260717)
        sensitivity = rng.normal(size=(64, 20))
        weights = rng.uniform(0.2, 2.0, size=64)
        values = np.abs(rng.normal(size=64))

        gram, eigenvalues = gram_and_eigenvalues(sensitivity, 1.3)
        alpha = 0.01 * np.trace(gram) / gram.shape[0]
        metrics = regularized_spectral_metrics(
            sensitivity, alpha, 1.3, gram, eigenvalues
        )
        selected, cumulative, _ = greedy_regularized_logdet(gram, alpha, 12)

        self.assertAlmostEqual(weighted_gini(values, weights), 0.43945735963931465)
        self.assertAlmostEqual(metrics["regularized_logdet"], 88.81848532460971)
        self.assertAlmostEqual(metrics["effective_rank_energy"], 17.05202151406474)
        self.assertEqual(selected, [5, 1, 16, 2, 14, 19, 13, 18, 17, 3, 0, 12])
        np.testing.assert_allclose(
            cumulative[-3:],
            [46.66026247641253, 51.1210280630329, 55.534869075168956],
            rtol=1e-13,
            atol=1e-13,
        )

    def test_common_grid_data_model(self) -> None:
        metas = [
            MeshMeta("A", "meshA.mat", 1, 4, 496, 80.0, 0.0, 315.0),
            MeshMeta("B", "meshB.mat", 1, 4, 496, 79.0, 1.0, 314.0),
        ]
        grid = make_common_grid(metas, nr=3, ntheta=4, nz=5)
        self.assertEqual(grid.shape, (5, 4, 3))
        self.assertEqual(grid.n_bins, 60)
        self.assertGreater(grid.bin_volume_mm3, 0.0)


if __name__ == "__main__":
    unittest.main()
