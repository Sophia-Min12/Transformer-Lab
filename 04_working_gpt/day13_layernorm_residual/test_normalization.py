"""Day 13 tests — normalization, the identity path, and what depth costs.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from normalization import (
    LayerNorm,
    Linear,
    PostNormResidual,
    PreNormResidual,
    Residual,
    Tensor,
    check_gradient,
    gradient_through_depth,
)


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_output_has_zero_mean_per_row(self):
        out = LayerNorm(6)(Tensor(self.rng.normal(size=(4, 6)) * 10 + 5))
        np.testing.assert_allclose(out.data.mean(axis=-1), np.zeros(4), atol=1e-9)

    def test_output_has_unit_variance_per_row(self):
        out = LayerNorm(6)(Tensor(self.rng.normal(size=(4, 6)) * 10 + 5))
        np.testing.assert_allclose(out.data.std(axis=-1), np.ones(4), atol=1e-4)

    def test_it_normalizes_per_position_not_per_batch(self):
        # Change one row; every other row's output must be untouched.
        x = self.rng.normal(size=(4, 6))
        before = LayerNorm(6)(Tensor(x)).data
        changed = x.copy()
        changed[2] *= 100.0
        after = LayerNorm(6)(Tensor(changed)).data
        np.testing.assert_allclose(before[[0, 1, 3]], after[[0, 1, 3]])

    def test_gamma_starts_at_one_and_beta_at_zero(self):
        layer = LayerNorm(5)
        np.testing.assert_array_equal(layer.gamma.data, np.ones(5))
        np.testing.assert_array_equal(layer.beta.data, np.zeros(5))

    def test_gamma_and_beta_are_applied(self):
        layer = LayerNorm(3)
        layer.gamma = Tensor([2.0, 2.0, 2.0])
        layer.beta = Tensor([1.0, 1.0, 1.0])
        out = layer(Tensor([[1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(float(out.data.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(out.data.std()), 2.0, places=3)

    def test_it_is_almost_invariant_to_shift_and_scale_of_the_input(self):
        # Shift invariance is exact - the mean is subtracted. Scale
        # invariance is not, because eps is a fixed absolute quantity
        # added to a variance that scales with the input: at 7x the
        # input, eps has 49x less relative weight. The gap here is ~2e-5,
        # the same species as Adam's eps on Day 7.
        x = self.rng.normal(size=(3, 8))
        base = LayerNorm(8)(Tensor(x)).data
        shifted_only = LayerNorm(8)(Tensor(x + 13.0)).data
        np.testing.assert_allclose(base, shifted_only, atol=1e-9)

        rescaled = LayerNorm(8)(Tensor(x * 7.0 + 13.0)).data
        np.testing.assert_allclose(base, rescaled, atol=1e-4)
        self.assertGreater(float(np.abs(base - rescaled).max()), 1e-9)

    def test_a_smaller_eps_tightens_the_scale_invariance(self):
        # Identifies eps as the cause rather than assuming it.
        x = self.rng.normal(size=(3, 8))
        gaps = []
        for eps in (1e-5, 1e-12):
            base = LayerNorm(8, eps=eps)(Tensor(x)).data
            rescaled = LayerNorm(8, eps=eps)(Tensor(x * 7.0)).data
            gaps.append(float(np.abs(base - rescaled).max()))
        self.assertGreater(gaps[0] / max(gaps[1], 1e-300), 100.0)

    def test_a_constant_row_stays_finite(self):
        # Variance is exactly zero; eps inside the root is what saves it.
        out = LayerNorm(4)(Tensor(np.full((2, 4), 3.0)))
        self.assertTrue(np.all(np.isfinite(out.data)))
        np.testing.assert_allclose(out.data, np.zeros((2, 4)))

    def test_a_constant_row_has_a_finite_gradient(self):
        x = Tensor(np.full((2, 4), 3.0))
        LayerNorm(4)(x).sum().backward()
        self.assertTrue(np.all(np.isfinite(x.grad)))

    def test_parameters(self):
        self.assertEqual(LayerNorm(16).parameter_count(), 32)

    def test_wrong_width_is_named(self):
        with self.assertRaises(ValueError):
            LayerNorm(4)(Tensor(np.zeros((2, 5))))

    def test_bad_arguments_rejected(self):
        with self.assertRaises(ValueError):
            LayerNorm(0)
        with self.assertRaises(ValueError):
            LayerNorm(4, eps=0.0)


class TestResidual(unittest.TestCase):
    def test_it_adds_the_input_back(self):
        layer = Residual(Linear(4, 4, seed=0))
        x = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        expected = x.data + layer.inner(x).data
        np.testing.assert_allclose(layer(x).data, expected)

    def test_a_zero_inner_layer_is_the_identity(self):
        inner = Linear(4, 4, seed=0)
        inner.weight = Tensor(np.zeros((4, 4)))
        x = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        np.testing.assert_allclose(Residual(inner)(x).data, x.data)

    def test_mismatched_shapes_rejected(self):
        with self.assertRaises(ValueError):
            Residual(Linear(4, 6, seed=0))(Tensor(np.zeros((3, 4))))

    def test_the_identity_path_carries_gradient(self):
        inner = Linear(4, 4, seed=0)
        inner.weight = Tensor(np.zeros((4, 4)))
        x = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        Residual(inner)(x).sum().backward()
        # f'(x) is zero, so all of the gradient came through the "1".
        np.testing.assert_allclose(x.grad, np.ones((3, 4)))


class TestDepth(unittest.TestCase):
    """The measurement the day exists for."""

    def test_a_plain_stack_loses_its_gradient(self):
        values = [gradient_through_depth(d, residual=False) for d in (1, 5, 10, 20, 40)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[-1], 1e-8)

    def test_a_residual_stack_keeps_it(self):
        self.assertGreater(gradient_through_depth(40, residual=True), 0.1)

    def test_the_gap_widens_with_depth(self):
        ratios = []
        for depth in (5, 20, 40):
            plain = gradient_through_depth(depth, residual=False)
            residual = gradient_through_depth(depth, residual=True)
            ratios.append(residual / plain)
        self.assertEqual(ratios, sorted(ratios))
        self.assertGreater(ratios[-1], 1e6)

    def test_residuals_can_inflate_rather_than_merely_preserve(self):
        # Honest caveat: the residual column grows with depth. Residuals
        # prevent vanishing; they do not by themselves prevent growth.
        shallow = gradient_through_depth(1, residual=True)
        deep = gradient_through_depth(40, residual=True)
        self.assertGreater(deep, shallow)


class TestPreNormVersusPostNorm(unittest.TestCase):
    @staticmethod
    def gradient_at_input(kind, depth=12, dim=16):
        rng = np.random.default_rng(0)
        x = Tensor(rng.normal(size=(6, dim)))
        h = x
        build = PreNormResidual if kind == "pre" else PostNormResidual
        for _ in range(depth):
            h = build(dim, Linear(dim, dim, seed=0))(h)
        h.sum().backward()
        return float(np.abs(x.grad).mean())

    def test_pre_norm_keeps_a_usable_gradient(self):
        self.assertGreater(self.gradient_at_input("pre"), 1e-3)

    def test_post_norm_loses_it(self):
        self.assertLess(self.gradient_at_input("post"), 1e-6)

    def test_they_differ_by_many_orders_of_magnitude(self):
        self.assertGreater(self.gradient_at_input("pre") / max(self.gradient_at_input("post"), 1e-300),
                           1e6)

    def test_both_preserve_shape(self):
        x = Tensor(np.zeros((4, 8)))
        for build in (PreNormResidual, PostNormResidual):
            self.assertEqual(build(8, Linear(8, 8, seed=0))(x).shape, (4, 8))

    def test_post_norm_output_is_normalized(self):
        # It ends with a LayerNorm, so its output has unit variance.
        x = Tensor(np.random.default_rng(0).normal(size=(4, 8)))
        out = PostNormResidual(8, Linear(8, 8, seed=0))(x)
        np.testing.assert_allclose(out.data.std(axis=-1), np.ones(4), atol=1e-3)

    def test_pre_norm_output_is_not(self):
        x = Tensor(np.random.default_rng(0).normal(size=(4, 8)) * 5)
        out = PreNormResidual(8, Linear(8, 8, seed=0))(x)
        self.assertFalse(np.allclose(out.data.std(axis=-1), np.ones(4), atol=1e-2))


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_layernorm(self):
        check_gradient(lambda t: (LayerNorm(5)(t) ** 2).sum(), [self.rng.normal(size=(3, 5))])

    def test_layernorm_on_a_batch_of_sequences(self):
        check_gradient(lambda t: (LayerNorm(4)(t) ** 2).sum(),
                       [self.rng.normal(size=(2, 3, 4))])

    def test_residual(self):
        check_gradient(lambda t, w: ((t + (t @ w).tanh()) ** 2).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4, 4))])

    def test_pre_norm_block(self):
        check_gradient(lambda t, w: ((t + (LayerNorm(4)(t) @ w).tanh()) ** 2).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4, 4))])


if __name__ == "__main__":
    unittest.main()
