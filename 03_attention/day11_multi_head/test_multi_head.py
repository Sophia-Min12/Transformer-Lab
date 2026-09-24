"""Day 11 tests — the split, the permutation, and the capacity claim.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from multi_head import (
    Adam,
    Linear,
    MultiHeadAttention,
    Tensor,
    check_gradient,
    merge_heads,
    permute,
    split_heads,
    two_relation_task,
)


class TestPermute(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(permute(Tensor(np.zeros((2, 3, 4))), (1, 0, 2)).shape, (3, 2, 4))

    def test_values_match_numpy(self):
        data = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_array_equal(permute(Tensor(data), (2, 0, 1)).data,
                                      data.transpose(2, 0, 1))

    def test_identity_permutation(self):
        data = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(permute(Tensor(data), (0, 1)).data, data)

    def test_gradient_returns_to_the_original_shape(self):
        x = Tensor(np.arange(24.0).reshape(2, 3, 4))
        (permute(x, (2, 1, 0)) ** 2).sum().backward()
        self.assertEqual(x.grad.shape, (2, 3, 4))

    def test_backward_moves_numbers_without_arithmetic(self):
        x = Tensor(np.zeros((2, 3)))
        permute(x, (1, 0)).sum().backward()
        np.testing.assert_array_equal(x.grad, np.ones((2, 3)))

    def test_bad_axes_rejected(self):
        for bad in ((0, 0), (0, 1, 2), (5, 1)):
            with self.assertRaises(ValueError):
                permute(Tensor(np.zeros((2, 3))), bad)


class TestSplitAndMerge(unittest.TestCase):
    def test_split_shape(self):
        self.assertEqual(split_heads(Tensor(np.zeros((5, 8))), 2).shape, (2, 5, 4))

    def test_one_head_is_a_leading_axis(self):
        self.assertEqual(split_heads(Tensor(np.zeros((5, 8))), 1).shape, (1, 5, 8))

    def test_merge_shape(self):
        self.assertEqual(merge_heads(Tensor(np.zeros((2, 5, 4)))).shape, (5, 8))

    def test_round_trip_is_exact(self):
        original = np.random.default_rng(0).normal(size=(6, 8))
        for heads in (1, 2, 4, 8):
            restored = merge_heads(split_heads(Tensor(original), heads)).data
            np.testing.assert_allclose(restored, original)

    def test_heads_get_disjoint_slices(self):
        data = np.arange(12.0).reshape(2, 6)
        split = split_heads(Tensor(data), 3).data
        np.testing.assert_array_equal(split[0, 0], [0.0, 1.0])
        np.testing.assert_array_equal(split[1, 0], [2.0, 3.0])
        np.testing.assert_array_equal(split[2, 0], [4.0, 5.0])

    def test_indivisible_dim_rejected(self):
        with self.assertRaises(ValueError):
            split_heads(Tensor(np.zeros((4, 6))), 4)

    def test_non_positive_heads_rejected(self):
        with self.assertRaises(ValueError):
            split_heads(Tensor(np.zeros((4, 6))), 0)


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape_is_unchanged_by_head_count(self):
        for heads in (1, 2, 4, 8):
            layer = MultiHeadAttention(dim=8, heads=heads, seed=0)
            out, weights = layer(Tensor(np.zeros((5, 8))))
            self.assertEqual(out.shape, (5, 8))
            self.assertEqual(weights.shape, (heads, 5, 5))

    def test_parameter_count_is_independent_of_heads(self):
        # Four d x d projections, however the width is sliced.
        counts = {MultiHeadAttention(dim=8, heads=h, seed=0).parameter_count()
                  for h in (1, 2, 4, 8)}
        self.assertEqual(counts, {4 * 8 * 8})

    def test_every_head_produces_a_distribution(self):
        layer = MultiHeadAttention(dim=8, heads=4, seed=0)
        _, weights = layer(Tensor(np.random.default_rng(0).normal(size=(6, 8))))
        np.testing.assert_allclose(weights.data.sum(axis=-1), np.ones((4, 6)))

    def test_causal_flag_makes_every_head_lower_triangular(self):
        layer = MultiHeadAttention(dim=8, heads=2, causal=True, seed=0)
        _, weights = layer(Tensor(np.random.default_rng(0).normal(size=(5, 8))))
        rows, cols = np.triu_indices(5, k=1)
        np.testing.assert_allclose(weights.data[:, rows, cols], 0.0, atol=1e-12)

    def test_every_projection_receives_gradient(self):
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        out, _ = layer(Tensor(np.random.default_rng(0).normal(size=(5, 8))))
        (out ** 2).sum().backward()
        for parameter in layer.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)

    def test_heads_are_genuinely_independent(self):
        # Different heads see different slices, so their weight matrices
        # must not be identical.
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        _, weights = layer(Tensor(np.random.default_rng(1).normal(size=(6, 8))))
        self.assertFalse(np.allclose(weights.data[0], weights.data[1]))

    def test_indivisible_dim_rejected(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(dim=6, heads=4)

    def test_bad_sizes_rejected(self):
        for args in ((0, 1), (8, 0)):
            with self.assertRaises(ValueError):
                MultiHeadAttention(dim=args[0], heads=args[1])


class TestTwoRelationTask(unittest.TestCase):
    def test_shapes(self):
        pairs = two_relation_task(sequences=3, length=6, dim=8, seed=0)
        self.assertEqual(len(pairs), 3)
        for x, target in pairs:
            self.assertEqual(x.shape, (6, 8))
            self.assertEqual(target.shape, (6, 2))

    def test_the_targets_are_the_flagged_lookups(self):
        x, target = two_relation_task(sequences=1, length=7, dim=8, seed=3)[0]
        first = int(np.argmax(x.data[:, 2]))
        second = int(np.argmax(x.data[:, 3]))
        np.testing.assert_allclose(target.data[:, 0], x.data[first, 0])
        np.testing.assert_allclose(target.data[:, 1], x.data[second, 1])

    def test_the_two_lookups_usually_point_at_different_positions(self):
        # If they coincided the task would need only one head.
        differ = 0
        for x, _ in two_relation_task(sequences=40, length=8, seed=1):
            if int(np.argmax(x.data[:, 2])) != int(np.argmax(x.data[:, 3])):
                differ += 1
        self.assertGreater(differ, 25)

    def test_deterministic(self):
        first = two_relation_task(sequences=2, seed=5)
        second = two_relation_task(sequences=2, seed=5)
        np.testing.assert_array_equal(first[0][0].data, second[0][0].data)


class TestCapacity(unittest.TestCase):
    """More heads fit the two-relation task better. Kept small to stay fast."""

    @staticmethod
    def fit(heads, pairs, steps=120):
        model = MultiHeadAttention(dim=8, heads=heads, seed=0)
        readout = Linear(8, 2, seed=99)
        optimizer = Adam(model.parameters() + readout.parameters(), lr=0.02)
        for _ in range(steps):
            optimizer.zero_grad()
            for sequence, wanted in pairs:
                ((readout(model(sequence)[0]) - wanted) ** 2).mean().backward()
            optimizer.step()
        return float(np.mean([
            float(((readout(model(s)[0]) - w) ** 2).mean().data) for s, w in pairs
        ]))

    def test_two_heads_fit_better_than_one(self):
        pairs = two_relation_task(sequences=40, length=8, seed=0)
        self.assertLess(self.fit(2, pairs), self.fit(1, pairs))

    def test_four_heads_fit_better_still(self):
        pairs = two_relation_task(sequences=40, length=8, seed=0)
        self.assertLess(self.fit(4, pairs), self.fit(2, pairs))


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_permute(self):
        check_gradient(lambda t: (permute(t, (1, 0, 2)) ** 2).sum(),
                       [self.rng.normal(size=(2, 3, 4))])

    def test_split_and_merge(self):
        check_gradient(lambda t: (merge_heads(split_heads(t, 2)) ** 2).sum(),
                       [self.rng.normal(size=(4, 6))])

    def test_a_full_multi_head_pass(self):
        layer = MultiHeadAttention(dim=6, heads=3, seed=0)
        out, _ = layer(Tensor(self.rng.normal(size=(4, 6))))
        (out ** 2).sum().backward()
        for parameter in layer.parameters():
            self.assertTrue(np.all(np.isfinite(parameter.grad)))


if __name__ == "__main__":
    unittest.main()
