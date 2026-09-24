"""Day 8 tests — the lookup, the scatter, and equivalence with one-hot.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from embeddings import (
    Embedding,
    Tensor,
    check_gradient,
    one_hot_matmul,
    take_rows,
)


class TestTakeRowsForward(unittest.TestCase):
    def setUp(self):
        self.table = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_selects_the_right_rows(self):
        np.testing.assert_allclose(take_rows(self.table, [2, 0]).data,
                                   [[5.0, 6.0], [1.0, 2.0]])

    def test_shape(self):
        self.assertEqual(take_rows(self.table, [0, 1, 2, 1]).shape, (4, 2))

    def test_a_repeated_index_gives_the_same_row(self):
        picked = take_rows(self.table, [1, 1]).data
        np.testing.assert_allclose(picked[0], picked[1])

    def test_empty_selection(self):
        self.assertEqual(take_rows(self.table, np.array([], dtype=int)).shape, (0, 2))

    def test_out_of_range_rejected(self):
        for bad in ([3], [-1]):
            with self.assertRaises(IndexError):
                take_rows(self.table, bad)

    def test_float_indices_rejected(self):
        with self.assertRaises(TypeError):
            take_rows(self.table, np.array([0.0, 1.0]))

    def test_non_2d_table_rejected(self):
        with self.assertRaises(ValueError):
            take_rows(Tensor([1.0, 2.0]), [0])


class TestScatterBackward(unittest.TestCase):
    """The one mistake that is easy to make here."""

    def test_a_row_read_three_times_gets_three_contributions(self):
        table = Tensor(np.zeros((4, 2)))
        take_rows(table, [1, 1, 1]).sum().backward()
        np.testing.assert_allclose(table.grad[1], [3.0, 3.0])

    def test_plain_fancy_index_assignment_would_give_one(self):
        # The wrong implementation, shown failing.
        wrong = np.zeros((4, 2))
        wrong[np.array([1, 1, 1])] += np.ones((3, 2))
        np.testing.assert_allclose(wrong[1], [1.0, 1.0])

    def test_untouched_rows_get_exactly_zero(self):
        table = Tensor(np.random.default_rng(0).normal(size=(5, 3)))
        (take_rows(table, [0, 1]) ** 2).sum().backward()
        for row in (2, 3, 4):
            np.testing.assert_array_equal(table.grad[row], np.zeros(3))

    def test_gradient_keeps_the_table_shape(self):
        table = Tensor(np.zeros((6, 3)))
        take_rows(table, [0, 5]).sum().backward()
        self.assertEqual(table.grad.shape, (6, 3))

    def test_mixed_repeats_accumulate_correctly(self):
        table = Tensor(np.zeros((3, 1)))
        take_rows(table, [0, 1, 1, 2, 2, 2]).sum().backward()
        np.testing.assert_allclose(table.grad.reshape(-1), [1.0, 2.0, 3.0])

    def test_gradient_checks(self):
        rng = np.random.default_rng(0)
        check_gradient(lambda t: (take_rows(t, [2, 0, 2]) ** 2).sum(),
                       [rng.normal(size=(4, 3))])
        check_gradient(lambda t: (take_rows(t, [1, 1, 1]) ** 2).sum(),
                       [rng.normal(size=(3, 2))])


class TestOneHotEquivalence(unittest.TestCase):
    """Same answer, same gradient, different cost."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.table = self.rng.normal(size=(10, 4))
        self.indices = [3, 7, 3, 0]

    def test_values_match(self):
        lookup = take_rows(Tensor(self.table), self.indices).data
        onehot = one_hot_matmul(Tensor(self.table), self.indices, 10).data
        np.testing.assert_allclose(lookup, onehot)

    def test_gradients_match(self):
        a, b = Tensor(self.table.copy()), Tensor(self.table.copy())
        (take_rows(a, self.indices) ** 2).sum().backward()
        (one_hot_matmul(b, self.indices, 10) ** 2).sum().backward()
        np.testing.assert_allclose(a.grad, b.grad, atol=1e-12)

    def test_the_one_hot_form_is_mostly_zeros(self):
        vocab, tokens = 1000, 8
        useful = tokens
        total = tokens * vocab
        self.assertLess(useful / total, 0.01)


class TestEmbeddingModule(unittest.TestCase):
    def test_output_shape_for_a_flat_index_list(self):
        self.assertEqual(Embedding(10, 4, seed=0)([1, 5, 1]).shape, (3, 4))

    def test_output_shape_for_a_batch_of_sequences(self):
        # (batch, time) -> (batch, time, dim), the shape Day 9 wants.
        indices = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(Embedding(10, 8, seed=0)(indices).shape, (2, 3, 8))

    def test_parameters(self):
        table = Embedding(20, 6, seed=0)
        self.assertEqual([p.shape for p in table.parameters()], [(20, 6)])
        self.assertEqual(table.parameter_count(), 120)

    def test_the_same_index_gives_the_same_vector(self):
        table = Embedding(10, 4, seed=0)
        out = table([3, 7, 3])
        np.testing.assert_allclose(out.data[0], out.data[2])

    def test_initial_scale_is_small(self):
        table = Embedding(500, 64, seed=0)
        self.assertLess(float(table.weight.data.std()), 0.05)

    def test_seed_is_reproducible(self):
        np.testing.assert_array_equal(Embedding(8, 3, seed=5).weight.data,
                                      Embedding(8, 3, seed=5).weight.data)

    def test_gradient_reaches_the_table(self):
        table = Embedding(10, 4, seed=0)
        (table([1, 2]) ** 2).sum().backward()
        self.assertGreater(float(np.abs(table.weight.grad).sum()), 0.0)

    def test_only_the_used_rows_move_under_training(self):
        table = Embedding(6, 3, seed=0)
        before = table.weight.data.copy()
        (table([0, 1]) ** 2).sum().backward()
        table.weight.data = table.weight.data - 0.1 * table.weight.grad
        for row in (2, 3, 4, 5):
            np.testing.assert_array_equal(table.weight.data[row], before[row])

    def test_bad_sizes_rejected(self):
        for args in ((0, 4), (4, 0)):
            with self.assertRaises(ValueError):
                Embedding(*args)

    def test_out_of_vocabulary_index_rejected(self):
        with self.assertRaises(IndexError):
            Embedding(5, 2, seed=0)([5])


if __name__ == "__main__":
    unittest.main()
