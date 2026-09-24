"""Day 10 tests — the mask, and the gradient proof that it works.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from masking import (
    MASK_VALUE,
    CausalSelfAttention,
    SelfAttention,
    Tensor,
    attention,
    causal_attention,
    causal_mask,
    check_gradient,
    depends_on,
    softmax,
)


class TestCausalMask(unittest.TestCase):
    def test_shape_and_values(self):
        np.testing.assert_allclose(causal_mask(3, value=-1.0),
                                   [[0, -1, -1], [0, 0, -1], [0, 0, 0]])

    def test_diagonal_is_open(self):
        np.testing.assert_array_equal(np.diag(causal_mask(5)), np.zeros(5))

    def test_lower_triangle_is_open(self):
        self.assertTrue(np.all(np.tril(causal_mask(6)) == 0.0))

    def test_upper_triangle_is_closed(self):
        mask = causal_mask(6)
        self.assertTrue(np.all(mask[np.triu_indices(6, k=1)] == MASK_VALUE))

    def test_length_one(self):
        np.testing.assert_allclose(causal_mask(1), [[0.0]])

    def test_non_positive_length_rejected(self):
        with self.assertRaises(ValueError):
            causal_mask(0)


class TestMaskedWeights(unittest.TestCase):
    def setUp(self):
        self.layer = CausalSelfAttention(dim=4, seed=0)
        _, self.weights = self.layer(Tensor(np.zeros((5, 4))))

    def test_rows_still_sum_to_one(self):
        # The mask removes options, not probability mass.
        np.testing.assert_allclose(self.weights.data.sum(axis=-1), np.ones(5))

    def test_future_weights_are_exactly_zero(self):
        upper = self.weights.data[np.triu_indices(5, k=1)]
        np.testing.assert_allclose(upper, np.zeros(upper.size), atol=1e-12)

    def test_row_t_spreads_over_t_plus_one_positions(self):
        # Identical inputs, so each allowed position gets an equal share.
        for row in range(5):
            np.testing.assert_allclose(self.weights.data[row, :row + 1],
                                       np.full(row + 1, 1 / (row + 1)))

    def test_the_first_row_sees_only_itself(self):
        self.assertAlmostEqual(float(self.weights.data[0, 0]), 1.0)


class TestCausalityByGradient(unittest.TestCase):
    """A zero weight is suggestive; a zero gradient is proof."""

    def test_dependence_is_strictly_lower_triangular(self):
        x = Tensor(np.random.default_rng(0).normal(size=(5, 4)))
        out, _ = CausalSelfAttention(dim=4, seed=0)(x)
        np.testing.assert_array_equal(depends_on(out, x),
                                      np.tril(np.ones((5, 5), dtype=bool)))

    def test_position_zero_depends_on_nothing_else(self):
        x = Tensor(np.random.default_rng(1).normal(size=(4, 4)))
        out, _ = CausalSelfAttention(dim=4, seed=0)(x)
        influence = depends_on(out, x)
        self.assertTrue(influence[0, 0])
        self.assertFalse(influence[0, 1:].any())

    def test_unmasked_attention_leaks_the_future(self):
        x = Tensor(np.random.default_rng(2).normal(size=(5, 4)))
        out, _ = SelfAttention(dim=4, seed=0)(x)
        self.assertTrue(depends_on(out, x).all())

    def test_the_probe_detects_a_mask_on_the_wrong_axis(self):
        # Transposing the mask still looks triangular in a printout while
        # letting the future in.
        rng = np.random.default_rng(3)
        q = k = v = Tensor(rng.normal(size=(4, 4)))
        out, _ = attention(q, k, v, mask=causal_mask(4).T)
        influence = depends_on(out, q)
        self.assertFalse(np.array_equal(influence, np.tril(np.ones((4, 4), bool))))


class TestMaskValueChoice(unittest.TestCase):
    """Why -1e9 rather than -inf."""

    def test_a_fully_masked_row_with_inf_gives_nan(self):
        scores = Tensor(np.zeros((1, 3)))
        with np.errstate(invalid="ignore"):
            result = softmax(scores + Tensor(np.full((1, 3), -np.inf)), axis=-1).data
        self.assertTrue(np.all(np.isnan(result)))

    def test_a_fully_masked_row_with_the_finite_value_is_uniform(self):
        scores = Tensor(np.zeros((1, 3)))
        result = softmax(scores + Tensor(np.full((1, 3), MASK_VALUE)), axis=-1).data
        np.testing.assert_allclose(result, np.full((1, 3), 1 / 3))

    def test_the_finite_value_still_zeroes_forbidden_entries(self):
        scores = Tensor(np.array([[1.0, 2.0]]))
        result = softmax(scores + Tensor(np.array([[0.0, MASK_VALUE]])), axis=-1).data
        np.testing.assert_allclose(result, [[1.0, 0.0]], atol=1e-12)

    def test_no_nan_anywhere_in_a_causal_forward_pass(self):
        rng = np.random.default_rng(0)
        out, weights = causal_attention(Tensor(rng.normal(size=(6, 4))),
                                        Tensor(rng.normal(size=(6, 4))),
                                        Tensor(rng.normal(size=(6, 4))))
        self.assertTrue(np.all(np.isfinite(out.data)))
        self.assertTrue(np.all(np.isfinite(weights.data)))


class TestCausalAttention(unittest.TestCase):
    def test_shapes(self):
        out, weights = causal_attention(Tensor(np.zeros((4, 3))),
                                        Tensor(np.zeros((4, 3))),
                                        Tensor(np.zeros((4, 3))))
        self.assertEqual((out.shape, weights.shape), ((4, 3), (4, 4)))

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            causal_attention(Tensor(np.zeros((3, 4))), Tensor(np.zeros((5, 4))),
                             Tensor(np.zeros((5, 4))))

    def test_the_first_output_equals_the_first_value(self):
        # Position 0 attends only to itself, so it copies value[0] exactly.
        rng = np.random.default_rng(0)
        value = rng.normal(size=(4, 3))
        out, _ = causal_attention(Tensor(rng.normal(size=(4, 5))),
                                  Tensor(rng.normal(size=(4, 5))), Tensor(value))
        np.testing.assert_allclose(out.data[0], value[0], atol=1e-9)

    def test_changing_a_later_value_leaves_earlier_outputs_alone(self):
        rng = np.random.default_rng(4)
        q = Tensor(rng.normal(size=(5, 4)))
        k = Tensor(rng.normal(size=(5, 4)))
        value = rng.normal(size=(5, 4))
        before, _ = causal_attention(q, k, Tensor(value))
        changed = value.copy()
        changed[4] += 100.0
        after, _ = causal_attention(q, k, Tensor(changed))
        np.testing.assert_allclose(before.data[:4], after.data[:4], atol=1e-9)
        self.assertFalse(np.allclose(before.data[4], after.data[4]))


class TestCausalSelfAttention(unittest.TestCase):
    def test_parameter_count_matches_unmasked(self):
        # The mask adds nothing to learn.
        self.assertEqual(CausalSelfAttention(dim=8, seed=0).parameter_count(),
                         SelfAttention(dim=8, seed=0).parameter_count())

    def test_every_projection_receives_gradient(self):
        layer = CausalSelfAttention(dim=6, seed=0)
        out, _ = layer(Tensor(np.random.default_rng(0).normal(size=(4, 6))))
        (out ** 2).sum().backward()
        for parameter in layer.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)

    def test_bad_dim_rejected(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(dim=0)


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_causal_attention(self):
        check_gradient(lambda a, b, c: causal_attention(a, b, c)[0].sum(),
                       [self.rng.normal(size=(4, 5)), self.rng.normal(size=(4, 5)),
                        self.rng.normal(size=(4, 5))])

    def test_squared(self):
        check_gradient(lambda a, b, c: (causal_attention(a, b, c)[0] ** 2).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(3, 4)),
                        self.rng.normal(size=(3, 4))])


if __name__ == "__main__":
    unittest.main()
