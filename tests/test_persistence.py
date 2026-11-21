"""Run from SihtiAI root: python -m unittest discover -s tests -v."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistence_head import sieve_features, choose_features, run_scene
from sihti.color import srgb_to_lab
from sihti.scenes import stripes
from sihti.graph import grid_graph


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = srgb_to_lab(stripes(H=24, W=32, noise=0)[0])

    def test_lossless_octaves(self):
        _, _, _, err = sieve_features(self.lab, J=4)
        self.assertLess(err, 1e-12)

    def test_sign_invariance(self):
        a = sieve_features(self.lab, J=3)
        b = sieve_features(-self.lab, J=3)
        np.testing.assert_allclose(a[2], b[2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a[1], -b[1], atol=1e-12, rtol=0)

    def test_unsigned_and_signed_are_distinct(self):
        core, signed, energy, _ = sieve_features(self.lab)
        a = choose_features("unsigned_res", self.lab, core, signed, energy, 1.)
        b = choose_features("signed_res", self.lab, core, signed, energy, 1.)
        self.assertNotEqual(a.shape[-1], b.shape[-1])
        self.assertTrue((energy >= 0).all())

    def test_spatial_scramble_preserves_marginal_energy(self):
        core, signed, energy, _ = sieve_features(self.lab)
        shuffled = choose_features("unsigned_scrambled", self.lab, core, signed, energy, 1.)
        original = choose_features("unsigned_res", self.lab, core, signed, energy, 1.)
        np.testing.assert_allclose(
            np.sort(shuffled[:, :, 3:].reshape(-1, energy.shape[-1]), axis=0),
            np.sort(original[:, :, 3:].reshape(-1, energy.shape[-1]), axis=0),
            atol=0, rtol=0)
        self.assertFalse(np.allclose(shuffled, original))

    def test_graph_affinity_symmetric(self):
        g = grid_graph(*self.lab.shape[:2], radius=2.)
        Wm = g.matrix(g.weights(self.lab, sigma=8., sigma_x=2.))
        self.assertEqual((Wm - Wm.T).nnz, 0)

    def test_small_end_to_end_without_gt_leak(self):
        out = run_scene("stripes", h=24, w=32, noise=0.,
                        J=2, methods=("self", "core_only", "signed_res", "unsigned_res"))
        self.assertLess(out["reconstruction_error"], 1e-12)
        for row in out["records"]:
            self.assertTrue(np.isfinite(row["ari"]))
            self.assertTrue(0 <= row["covering"] <= 1)

if __name__ == "__main__":
    unittest.main()
