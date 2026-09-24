"""Day 14 tests — the position-wise network, GELU, and the assembled block.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from block import (
    FeedForward,
    Tensor,
    TransformerBlock,
    check_gradient,
    gelu,
    parameter_split,
)


class TestGelu(unittest.TestCase):
    def test_zero_maps_to_zero(self):
        self.assertAlmostEqual(float(gelu(Tensor([0.0])).data[0]), 0.0)

    def test_large_positive_is_nearly_the_identity(self):
        self.assertAlmostEqual(float(gelu(Tensor([5.0])).data[0]), 5.0, places=4)

    def test_large_negative_is_nearly_zero(self):
        self.assertAlmostEqual(float(gelu(Tensor([-5.0])).data[0]), 0.0, places=4)

    def test_it_dips_below_zero(self):
        # The property that distinguishes it from ReLU.
        self.assertLess(float(gelu(Tensor([-1.0])).data[0]), 0.0)

    def test_it_is_monotone_for_positive_input(self):
        values = [float(gelu(Tensor([x])).data[0]) for x in (0.5, 1.0, 2.0, 4.0)]
        self.assertEqual(values, sorted(values))

    def test_relu_has_no_gradient_at_a_negative_input_and_gelu_does(self):
        relu_point, gelu_point = Tensor([-1.0]), Tensor([-1.0])
        relu_point.relu().sum().backward()
        gelu(gelu_point).sum().backward()
        self.assertEqual(float(relu_point.grad[0]), 0.0)
        self.assertNotEqual(float(gelu_point.grad[0]), 0.0)

    def test_it_approximates_the_gaussian_cdf_form(self):
        # 0.5 x (1 + erf(x / sqrt(2))) is the exact definition.
        for x in (-2.0, -0.5, 0.5, 2.0):
            exact = 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
            self.assertAlmostEqual(float(gelu(Tensor([x])).data[0]), exact, places=3)


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_shape_is_preserved(self):
        self.assertEqual(FeedForward(dim=8, seed=0)(Tensor(np.zeros((5, 8)))).shape, (5, 8))

    def test_default_hidden_is_four_times_the_width(self):
        self.assertEqual(FeedForward(dim=16, seed=0).hidden, 64)

    def test_it_acts_on_each_position_independently(self):
        # The defining property: no mixing across time.
        layer = FeedForward(dim=8, seed=0)
        x = self.rng.normal(size=(5, 8))
        before = layer(Tensor(x)).data
        changed = x.copy()
        changed[3] += 10.0
        after = layer(Tensor(changed)).data
        np.testing.assert_allclose(before[[0, 1, 2, 4]], after[[0, 1, 2, 4]])

    def test_the_changed_position_does_change(self):
        layer = FeedForward(dim=8, seed=0)
        x = self.rng.normal(size=(5, 8))
        changed = x.copy()
        changed[3] += 10.0
        self.assertFalse(np.allclose(layer(Tensor(x)).data[3],
                                     layer(Tensor(changed)).data[3]))

    def test_parameter_count(self):
        layer = FeedForward(dim=8, hidden=32, seed=0)
        self.assertEqual(layer.parameter_count(), 8 * 32 + 32 + 32 * 8 + 8)

    def test_extra_leading_dimensions_pass_through(self):
        self.assertEqual(FeedForward(dim=4, seed=0)(Tensor(np.zeros((2, 3, 4)))).shape,
                         (2, 3, 4))

    def test_all_three_activations_run(self):
        for activation in ("gelu", "relu", "tanh"):
            layer = FeedForward(dim=4, activation=activation, seed=0)
            self.assertEqual(layer(Tensor(np.zeros((2, 4)))).shape, (2, 4))

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            FeedForward(dim=4, activation="swish")

    def test_bad_dim_rejected(self):
        with self.assertRaises(ValueError):
            FeedForward(dim=0)


class TestWhereTheParametersAre(unittest.TestCase):
    """Two thirds of a transformer is the feed-forward layer."""

    def test_feed_forward_is_twice_attention(self):
        for dim in (64, 256, 768):
            split = parameter_split(dim, heads=8)
            self.assertAlmostEqual(split["feedforward"] / split["attention"], 2.0, delta=0.05)

    def test_the_share_is_about_two_thirds_at_every_width(self):
        for dim in (64, 256, 768):
            split = parameter_split(dim, heads=8)
            self.assertAlmostEqual(split["feedforward"] / split["total"], 0.67, delta=0.02)

    def test_layernorms_are_negligible(self):
        split = parameter_split(768, heads=8)
        self.assertLess(split["layernorms"] / split["total"], 0.001)

    def test_attention_is_four_d_squared(self):
        split = parameter_split(64, heads=8)
        self.assertEqual(split["attention"], 4 * 64 * 64)

    def test_the_parts_sum_to_the_total(self):
        split = parameter_split(128, heads=4)
        self.assertEqual(split["attention"] + split["feedforward"] + split["layernorms"],
                         split["total"])


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_shape_is_preserved(self):
        block = TransformerBlock(dim=8, heads=2, seed=0)
        self.assertEqual(block(Tensor(np.zeros((5, 8)))).shape, (5, 8))

    def test_blocks_stack(self):
        x = Tensor(self.rng.normal(size=(6, 8)))
        for index in range(4):
            x = TransformerBlock(dim=8, heads=2, seed=index)(x)
        self.assertEqual(x.shape, (6, 8))

    def test_it_stays_causal(self):
        block = TransformerBlock(dim=8, heads=2, causal=True, seed=0)
        weights = block.attention_weights(Tensor(self.rng.normal(size=(6, 8))))
        rows, cols = np.triu_indices(6, k=1)
        np.testing.assert_allclose(weights.data[:, rows, cols], 0.0, atol=1e-12)

    def test_causality_survives_a_stack(self):
        # Each block is causal, so the composition must be.
        x = Tensor(self.rng.normal(size=(5, 8)))
        h = x
        for index in range(3):
            h = TransformerBlock(dim=8, heads=2, seed=index)(h)
        for row in range(5):
            x.grad = np.zeros_like(x.data)
            selector = np.zeros((5, 8))
            selector[row] = 1.0
            (h * Tensor(selector)).sum().backward()
            influence = np.abs(x.grad).sum(axis=-1) > 1e-12
            self.assertFalse(influence[row + 1:].any(), f"row {row} saw the future")

    def test_every_parameter_receives_gradient(self):
        block = TransformerBlock(dim=6, heads=2, seed=0)
        (block(Tensor(self.rng.normal(size=(4, 6)))) ** 2).sum().backward()
        for parameter in block.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)
            self.assertTrue(np.all(np.isfinite(parameter.grad)))

    def test_a_deep_stack_keeps_its_gradient(self):
        # Day 13's residuals, doing their job inside a real block.
        x = Tensor(self.rng.normal(size=(6, 8)))
        h = x
        for index in range(12):
            h = TransformerBlock(dim=8, heads=2, seed=index)(h)
        h.sum().backward()
        self.assertGreater(float(np.abs(x.grad).mean()), 0.1)

    def test_parameter_count_matches_its_parts(self):
        block = TransformerBlock(dim=16, heads=4, seed=0)
        parts = (block.attention.parameter_count() + block.feedforward.parameter_count()
                 + block.attention_norm.parameter_count()
                 + block.feedforward_norm.parameter_count())
        self.assertEqual(block.parameter_count(), parts)


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_gelu(self):
        check_gradient(lambda t: gelu(t).sum(), [self.rng.normal(size=(4,))])

    def test_gelu_squared(self):
        check_gradient(lambda t: (gelu(t) ** 2).sum(), [self.rng.normal(size=(3, 4))])

    def test_feed_forward(self):
        check_gradient(
            lambda t, w1, b1, w2, b2: ((gelu(t @ w1 + b1) @ w2 + b2) ** 2).sum(),
            [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4, 8)),
             self.rng.normal(size=(8,)), self.rng.normal(size=(8, 4)),
             self.rng.normal(size=(4,))],
        )


if __name__ == "__main__":
    unittest.main()
