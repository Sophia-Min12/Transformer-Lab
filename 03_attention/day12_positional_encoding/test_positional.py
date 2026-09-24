"""Day 12 tests — equivariance, the sinusoids, and the two encodings.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from positional import (
    LearnedPositionalEncoding,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    Tensor,
    check_gradient,
    permutation_gap,
    sinusoidal_encoding,
    take_rows,
)


class TestPermutationEquivariance(unittest.TestCase):
    """The problem the day exists to solve."""

    def setUp(self):
        self.x = Tensor(np.random.default_rng(0).normal(size=(6, 8)))
        self.order = np.array([3, 1, 5, 0, 4, 2])

    def test_plain_attention_cannot_tell_orderings_apart(self):
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        self.assertLess(permutation_gap(layer, self.x, self.order), 1e-12)

    def test_the_identity_permutation_always_gives_zero(self):
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        self.assertLess(permutation_gap(layer, self.x, np.arange(6)), 1e-12)

    def test_causal_masking_does_break_the_symmetry(self):
        # Contrary to my first draft. Position t sees exactly {0..t}, so
        # shuffling changes which tokens fall inside that set.
        layer = MultiHeadAttention(dim=8, heads=2, causal=True, seed=0)
        self.assertGreater(permutation_gap(layer, self.x, self.order), 1e-3)

    def test_a_positional_encoding_breaks_it_too(self):
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        encoder = SinusoidalPositionalEncoding(dim=8)
        self.assertGreater(permutation_gap(lambda t: layer(encoder(t)), self.x, self.order),
                           1e-3)

    def test_a_learned_encoding_breaks_it_as_well(self):
        layer = MultiHeadAttention(dim=8, heads=2, seed=0)
        encoder = LearnedPositionalEncoding(dim=8, max_length=16, seed=0)
        self.assertGreater(permutation_gap(lambda t: layer(encoder(t)), self.x, self.order),
                           1e-6)


class TestSinusoidalTable(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(sinusoidal_encoding(10, 16).shape, (10, 16))

    def test_position_zero(self):
        # sin(0)=0, cos(0)=1, alternating.
        np.testing.assert_allclose(sinusoidal_encoding(1, 4)[0], [0.0, 1.0, 0.0, 1.0])

    def test_values_are_bounded(self):
        table = sinusoidal_encoding(256, 32)
        self.assertGreaterEqual(table.min(), -1.0)
        self.assertLessEqual(table.max(), 1.0)

    def test_every_position_is_distinct(self):
        table = sinusoidal_encoding(128, 32)
        for step in (1, 2, 17, 64):
            self.assertGreater(float(np.linalg.norm(table[0] - table[step])), 1e-6)

    def test_all_positions_have_the_same_norm(self):
        # sin^2 + cos^2 = 1 per pair, so the norm is sqrt(dim/2) everywhere
        # and no position shouts louder than another.
        norms = np.linalg.norm(sinusoidal_encoding(128, 64), axis=1)
        np.testing.assert_allclose(norms, np.full(128, math.sqrt(32)))

    def test_distance_grows_then_plateaus(self):
        table = sinusoidal_encoding(512, 64)
        distances = [float(np.linalg.norm(table[0] - table[k])) for k in (1, 2, 8, 64, 256)]
        self.assertEqual(distances, sorted(distances))
        self.assertLess(distances[-1] / distances[-2], 1.5)

    def test_relative_offset_is_a_fixed_rotation(self):
        table = sinusoidal_encoding(64, 16)
        pair = table[:, :2]
        angles = []
        for base in (5, 20, 40):
            before, after = pair[base], pair[base + 3]
            raw = math.atan2(after[0], after[1]) - math.atan2(before[0], before[1])
            angles.append(math.atan2(math.sin(raw), math.cos(raw)))
        for angle in angles[1:]:
            self.assertAlmostEqual(angle, angles[0], places=9)

    def test_odd_dim_rejected(self):
        with self.assertRaises(ValueError):
            sinusoidal_encoding(4, 5)

    def test_non_positive_rejected(self):
        for args in ((0, 4), (4, 0)):
            with self.assertRaises(ValueError):
                sinusoidal_encoding(*args)


class TestSinusoidalModule(unittest.TestCase):
    def test_shape_is_preserved(self):
        self.assertEqual(SinusoidalPositionalEncoding(dim=4)(Tensor(np.zeros((3, 4)))).shape,
                         (3, 4))

    def test_it_has_no_parameters(self):
        self.assertEqual(SinusoidalPositionalEncoding(dim=8).parameters(), [])

    def test_it_adds_the_table(self):
        layer = SinusoidalPositionalEncoding(dim=4)
        out = layer(Tensor(np.zeros((3, 4)))).data
        np.testing.assert_allclose(out, sinusoidal_encoding(3, 4))

    def test_it_extends_past_the_initial_table(self):
        layer = SinusoidalPositionalEncoding(dim=8, max_length=4)
        self.assertEqual(layer(Tensor(np.zeros((50, 8)))).shape, (50, 8))

    def test_it_handles_a_very_long_sequence(self):
        layer = SinusoidalPositionalEncoding(dim=8)
        self.assertEqual(layer(Tensor(np.zeros((5000, 8)))).shape, (5000, 8))

    def test_gradient_passes_through_unchanged(self):
        x = Tensor(np.random.default_rng(0).normal(size=(4, 8)))
        SinusoidalPositionalEncoding(dim=8)(x).sum().backward()
        np.testing.assert_allclose(x.grad, np.ones((4, 8)))


class TestLearnedEncoding(unittest.TestCase):
    def test_shape_is_preserved(self):
        layer = LearnedPositionalEncoding(dim=4, max_length=8, seed=0)
        self.assertEqual(layer(Tensor(np.zeros((3, 4)))).shape, (3, 4))

    def test_it_has_one_row_per_position(self):
        layer = LearnedPositionalEncoding(dim=6, max_length=32, seed=0)
        self.assertEqual(layer.parameter_count(), 32 * 6)

    def test_beyond_max_length_raises(self):
        layer = LearnedPositionalEncoding(dim=4, max_length=4, seed=0)
        with self.assertRaises(IndexError):
            layer(Tensor(np.zeros((5, 4))))

    def test_only_the_used_rows_receive_gradient(self):
        layer = LearnedPositionalEncoding(dim=4, max_length=8, seed=0)
        layer(Tensor(np.zeros((3, 4)))).sum().backward()
        self.assertGreater(float(np.abs(layer.weight.grad[:3]).sum()), 0.0)
        np.testing.assert_array_equal(layer.weight.grad[3:], np.zeros((5, 4)))

    def test_bad_sizes_rejected(self):
        for args in ((0, 8), (4, 0)):
            with self.assertRaises(ValueError):
                LearnedPositionalEncoding(dim=args[0], max_length=args[1])


class TestTheTradeBetweenThem(unittest.TestCase):
    def test_sinusoidal_costs_no_parameters(self):
        self.assertEqual(SinusoidalPositionalEncoding(dim=64).parameter_count(), 0)

    def test_learned_costs_one_vector_per_position(self):
        self.assertGreater(LearnedPositionalEncoding(dim=64, max_length=512,
                                                     seed=0).parameter_count(), 0)

    def test_sinusoidal_extrapolates_where_learned_cannot(self):
        length = 600
        self.assertEqual(
            SinusoidalPositionalEncoding(dim=8)(Tensor(np.zeros((length, 8)))).shape,
            (length, 8),
        )
        with self.assertRaises(IndexError):
            LearnedPositionalEncoding(dim=8, max_length=512, seed=0)(
                Tensor(np.zeros((length, 8))))


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_sinusoidal(self):
        check_gradient(lambda t: (SinusoidalPositionalEncoding(4)(t) ** 2).sum(),
                       [self.rng.normal(size=(5, 4))])

    def test_learned(self):
        check_gradient(lambda t, w: ((t + take_rows(w, np.arange(3))) ** 2).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(6, 4))])


if __name__ == "__main__":
    unittest.main()
