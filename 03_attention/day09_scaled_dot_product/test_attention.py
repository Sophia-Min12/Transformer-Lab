"""Day 9 tests — the distribution, the scale, and what it protects.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from attention import (
    SelfAttention,
    Tensor,
    attention,
    attention_entropy,
    attention_scores,
    check_gradient,
)


def _softmax_sensitivity(weights) -> float:
    """Mean of sum_j p_j(1 - p_j): how much the weights move per unit score."""
    p = np.asarray(weights.data)
    return float(np.mean(np.sum(p * (1 - p), axis=-1)))


class TestAttentionScores(unittest.TestCase):
    def test_shape(self):
        scores = attention_scores(Tensor(np.zeros((3, 4))), Tensor(np.zeros((5, 4))))
        self.assertEqual(scores.shape, (3, 5))

    def test_worked_example(self):
        q = Tensor([[1.0, 0.0]])
        k = Tensor([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(attention_scores(q, k).data,
                                   [[1 / math.sqrt(2), 0.0]])

    def test_unscaled_is_the_plain_dot_product(self):
        q = Tensor([[1.0, 2.0]])
        k = Tensor([[3.0, 4.0]])
        np.testing.assert_allclose(attention_scores(q, k, scale=False).data, [[11.0]])

    def test_scale_divides_by_sqrt_dk(self):
        q = Tensor(np.ones((1, 16)))
        k = Tensor(np.ones((1, 16)))
        raw = attention_scores(q, k, scale=False).data
        scaled = attention_scores(q, k, scale=True).data
        np.testing.assert_allclose(scaled, raw / 4.0)

    def test_mismatched_last_dimension_rejected(self):
        with self.assertRaises(ValueError):
            attention_scores(Tensor(np.zeros((3, 4))), Tensor(np.zeros((3, 5))))


class TestAttentionOutput(unittest.TestCase):
    def test_weights_are_a_distribution(self):
        rng = np.random.default_rng(0)
        _, weights = attention(Tensor(rng.normal(size=(4, 8))),
                               Tensor(rng.normal(size=(6, 8))),
                               Tensor(rng.normal(size=(6, 8))))
        np.testing.assert_allclose(weights.data.sum(axis=-1), np.ones(4))
        self.assertGreaterEqual(weights.data.min(), 0.0)

    def test_output_shape_follows_value(self):
        out, _ = attention(Tensor(np.zeros((3, 4))),
                           Tensor(np.zeros((5, 4))),
                           Tensor(np.zeros((5, 7))))
        self.assertEqual(out.shape, (3, 7))

    def test_identical_keys_give_uniform_attention(self):
        q = Tensor(np.ones((1, 4)))
        k = Tensor(np.ones((3, 4)))
        v = Tensor(np.eye(3))
        _, weights = attention(q, k, v)
        np.testing.assert_allclose(weights.data, np.full((1, 3), 1 / 3))

    def test_output_is_a_convex_combination_of_values(self):
        # Every output must lie inside the range of the values it mixes.
        rng = np.random.default_rng(1)
        value = rng.normal(size=(6, 3))
        out, _ = attention(Tensor(rng.normal(size=(4, 8))),
                           Tensor(rng.normal(size=(6, 8))), Tensor(value))
        self.assertGreaterEqual(out.data.min(), value.min() - 1e-9)
        self.assertLessEqual(out.data.max(), value.max() + 1e-9)

    def test_a_dominant_score_selects_one_value(self):
        q = Tensor([[100.0, 0.0]])
        k = Tensor([[1.0, 0.0], [0.0, 1.0]])
        v = Tensor([[7.0], [9.0]])
        out, _ = attention(q, k, v, scale=False)
        self.assertAlmostEqual(float(out.data[0, 0]), 7.0, places=6)

    def test_key_and_value_length_must_match(self):
        with self.assertRaises(ValueError):
            attention(Tensor(np.zeros((2, 4))), Tensor(np.zeros((5, 4))),
                      Tensor(np.zeros((6, 4))))

    def test_a_mask_of_zeros_changes_nothing(self):
        rng = np.random.default_rng(2)
        q, k, v = (Tensor(rng.normal(size=(3, 4))) for _ in range(3))
        plain, _ = attention(q, k, v)
        masked, _ = attention(q, k, v, mask=np.zeros((3, 3)))
        np.testing.assert_allclose(plain.data, masked.data)

    def test_a_badly_shaped_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            attention(Tensor(np.zeros((3, 4))), Tensor(np.zeros((3, 4))),
                      Tensor(np.zeros((3, 4))), mask=np.zeros((2, 2)))


class TestWhyTheScaleExists(unittest.TestCase):
    """Three consequences, each measured."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def scores_std(self, d_k, scale):
        q = Tensor(self.rng.normal(size=(64, d_k)))
        k = Tensor(self.rng.normal(size=(64, d_k)))
        return float(attention_scores(q, k, scale=scale).data.std())

    def test_unscaled_spread_grows_as_sqrt_dk(self):
        for d_k in (16, 64, 256, 1024):
            self.assertAlmostEqual(self.scores_std(d_k, False) / math.sqrt(d_k),
                                   1.0, delta=0.15)

    def test_scaled_spread_is_constant(self):
        for d_k in (4, 16, 64, 256, 1024):
            self.assertAlmostEqual(self.scores_std(d_k, True), 1.0, delta=0.15)

    def weights_at(self, d_k, scale):
        q = Tensor(self.rng.normal(size=(64, d_k)))
        k = Tensor(self.rng.normal(size=(64, d_k)))
        v = Tensor(self.rng.normal(size=(64, d_k)))
        return attention(q, k, v, scale=scale)[1]

    def test_unscaled_entropy_collapses_with_dimension(self):
        entropies = [float(attention_entropy(self.weights_at(d, False)).mean())
                     for d in (4, 16, 64, 256, 1024)]
        self.assertEqual(entropies, sorted(entropies, reverse=True))
        self.assertLess(entropies[-1], 0.5)

    def test_scaled_entropy_stays_high(self):
        for d_k in (4, 64, 1024):
            entropy = float(attention_entropy(self.weights_at(d_k, True)).mean())
            self.assertGreater(entropy, 0.7 * math.log(64))

    def test_unscaled_softmax_sensitivity_collapses(self):
        # The actual claim: a saturated softmax stops responding to its
        # input, so the query and key projections receive no signal.
        values = [_softmax_sensitivity(self.weights_at(d, False))
                  for d in (4, 16, 64, 256, 1024)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0] / values[-1], 5.0)

    def test_scaled_sensitivity_is_flat(self):
        values = [_softmax_sensitivity(self.weights_at(d, True))
                  for d in (4, 64, 1024)]
        self.assertLess(max(values) - min(values), 0.1)

    def test_the_naive_gradient_probe_reads_backwards(self):
        # Worth pinning: |d out / d query| is LARGER unscaled, because the
        # 1/sqrt(d_k) factor multiplies that gradient directly. It
        # conflates the scale with the saturation and must not be used as
        # evidence.
        sizes = []
        for scale in (False, True):
            q = Tensor(self.rng.normal(size=(32, 256)))
            k = Tensor(self.rng.normal(size=(32, 256)))
            v = Tensor(self.rng.normal(size=(32, 256)))
            out, _ = attention(q, k, v, scale=scale)
            out.sum().backward()
            sizes.append(float(np.abs(q.grad).mean()))
        self.assertGreater(sizes[0], sizes[1])


class TestAttentionEntropy(unittest.TestCase):
    def test_uniform_is_log_n(self):
        uniform = Tensor(np.full((1, 8), 1 / 8))
        self.assertAlmostEqual(float(attention_entropy(uniform)[0]), math.log(8))

    def test_one_hot_is_zero(self):
        peaked = Tensor(np.array([[1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(attention_entropy(peaked)[0]), 0.0)

    def test_never_negative(self):
        rng = np.random.default_rng(0)
        _, weights = attention(Tensor(rng.normal(size=(5, 4))),
                               Tensor(rng.normal(size=(5, 4))),
                               Tensor(rng.normal(size=(5, 4))))
        self.assertTrue(np.all(attention_entropy(weights) >= 0.0))


class TestSelfAttention(unittest.TestCase):
    def test_shapes(self):
        layer = SelfAttention(dim=8, seed=0)
        out, weights = layer(Tensor(np.zeros((5, 8))))
        self.assertEqual(out.shape, (5, 8))
        self.assertEqual(weights.shape, (5, 5))

    def test_all_parameters_are_in_the_projections(self):
        layer = SelfAttention(dim=8, seed=0)
        self.assertEqual(layer.parameter_count(), 3 * 8 * 8)

    def test_every_projection_receives_gradient(self):
        layer = SelfAttention(dim=6, seed=0)
        out, _ = layer(Tensor(np.random.default_rng(0).normal(size=(4, 6))))
        (out ** 2).sum().backward()
        for parameter in layer.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)

    def test_every_position_can_see_every_other(self):
        # Day 10 is about removing exactly this.
        layer = SelfAttention(dim=4, seed=0)
        _, weights = layer(Tensor(np.random.default_rng(0).normal(size=(5, 4))))
        self.assertTrue(np.all(weights.data > 0.0))

    def test_a_separate_head_dimension(self):
        layer = SelfAttention(dim=8, head_dim=3, seed=0)
        out, _ = layer(Tensor(np.zeros((4, 8))))
        self.assertEqual(out.shape, (4, 3))

    def test_bad_dim_rejected(self):
        with self.assertRaises(ValueError):
            SelfAttention(dim=0)


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_scores(self):
        check_gradient(lambda a, b: attention_scores(a, b).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(5, 4))])

    def test_full_attention(self):
        check_gradient(lambda a, b, c: attention(a, b, c)[0].sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(5, 4)),
                        self.rng.normal(size=(5, 4))])

    def test_attention_squared(self):
        check_gradient(lambda a, b, c: (attention(a, b, c)[0] ** 2).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(5, 4)),
                        self.rng.normal(size=(5, 4))])

    def test_self_attention_projections(self):
        check_gradient(lambda x, wq, wk, wv: attention(x @ wq, x @ wk, x @ wv)[0].sum(),
                       [self.rng.normal(size=(4, 6)), self.rng.normal(size=(6, 6)),
                        self.rng.normal(size=(6, 6)), self.rng.normal(size=(6, 6))])

    def test_with_a_mask(self):
        mask = np.triu(np.full((4, 4), -1e9), k=1)
        check_gradient(lambda a, b, c: attention(a, b, c, mask=mask)[0].sum(),
                       [self.rng.normal(size=(4, 5)), self.rng.normal(size=(4, 5)),
                        self.rng.normal(size=(4, 5))])


if __name__ == "__main__":
    unittest.main()
