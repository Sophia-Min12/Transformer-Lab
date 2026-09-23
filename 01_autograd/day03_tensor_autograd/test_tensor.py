"""Day 3 tests — shapes, broadcasting, and the bug that hides inside it.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from tensor import Tensor, check_gradient, numerical_gradient, unbroadcast


class TestUnbroadcast(unittest.TestCase):
    def test_sums_away_a_prepended_axis(self):
        np.testing.assert_array_equal(unbroadcast(np.ones((2, 3)), (3,)), [2.0, 2.0, 2.0])

    def test_sums_a_stretched_axis_keeping_it(self):
        np.testing.assert_array_equal(unbroadcast(np.ones((2, 3)), (1, 3)), [[2.0, 2.0, 2.0]])

    def test_sums_both_kinds_at_once(self):
        result = unbroadcast(np.ones((4, 2, 3)), (1, 3))
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_array_equal(result, [[8.0, 8.0, 8.0]])

    def test_matching_shape_is_unchanged(self):
        gradient = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(unbroadcast(gradient, (2, 3)), gradient)

    def test_reduces_all_the_way_to_a_scalar(self):
        self.assertEqual(unbroadcast(np.ones((2, 3)), ()).shape, ())
        self.assertEqual(float(unbroadcast(np.ones((2, 3)), ())), 6.0)

    def test_total_is_conserved(self):
        # Summing cannot invent or lose gradient mass.
        gradient = np.arange(12.0).reshape(4, 3)
        for shape in ((3,), (1, 3), (4, 1), (4, 3)):
            self.assertAlmostEqual(float(unbroadcast(gradient, shape).sum()), float(gradient.sum()))


class TestTensorBasics(unittest.TestCase):
    def test_accepts_lists_and_arrays(self):
        self.assertEqual(Tensor([1.0, 2.0]).shape, (2,))
        self.assertEqual(Tensor(np.zeros((2, 3))).shape, (2, 3))

    def test_stores_as_float(self):
        self.assertEqual(Tensor([1, 2]).data.dtype, np.float64)

    def test_grad_starts_at_zeros_of_the_same_shape(self):
        t = Tensor(np.zeros((2, 3)))
        self.assertEqual(t.grad.shape, (2, 3))
        self.assertEqual(float(t.grad.sum()), 0.0)

    def test_rejects_a_tensor_as_data(self):
        with self.assertRaises(TypeError):
            Tensor(Tensor([1.0]))

    def test_repr_shows_shape(self):
        self.assertIn("shape=(2,)", repr(Tensor([1.0, 2.0])))


class TestForwardShapes(unittest.TestCase):
    def test_elementwise_ops_keep_shape(self):
        a = Tensor(np.ones((2, 3)))
        for out in (a + a, a * a, a ** 2, a.tanh(), a.relu(), a.exp()):
            self.assertEqual(out.shape, (2, 3))

    def test_broadcast_add(self):
        self.assertEqual((Tensor(np.ones((2, 3))) + Tensor([1.0, 2.0, 3.0])).shape, (2, 3))

    def test_sum_to_scalar(self):
        self.assertEqual(Tensor(np.ones((2, 3))).sum().shape, ())

    def test_sum_along_an_axis(self):
        self.assertEqual(Tensor(np.ones((2, 3))).sum(axis=0).shape, (3,))

    def test_sum_keepdims(self):
        self.assertEqual(Tensor(np.ones((2, 3))).sum(axis=0, keepdims=True).shape, (1, 3))

    def test_mean_value(self):
        self.assertAlmostEqual(float(Tensor([1.0, 2.0, 3.0]).mean().data), 2.0)

    def test_reshape(self):
        self.assertEqual(Tensor(np.ones((2, 3))).reshape(3, 2).shape, (3, 2))

    def test_reshape_accepts_a_tuple(self):
        self.assertEqual(Tensor(np.ones((2, 3))).reshape((6,)).shape, (6,))

    def test_transpose(self):
        self.assertEqual(Tensor(np.ones((2, 3))).T.shape, (3, 2))

    def test_transpose_needs_two_dimensions(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0]).T

    def test_log_of_non_positive_raises(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, -1.0]).log()


class TestBroadcastGradient(unittest.TestCase):
    """The subtle bug this day exists for."""

    def test_a_broadcast_row_accumulates_once_per_use(self):
        rows = Tensor(np.ones((2, 3)))
        bias = Tensor([10.0, 20.0, 30.0])
        (rows + bias).sum().backward()
        # Each bias entry fed two output entries.
        np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(rows.grad, np.ones((2, 3)))

    def test_gradients_keep_the_input_shape(self):
        rows = Tensor(np.ones((4, 3)))
        bias = Tensor([1.0, 2.0, 3.0])
        (rows * bias).sum().backward()
        self.assertEqual(bias.grad.shape, (3,))
        self.assertEqual(rows.grad.shape, (4, 3))

    def test_a_size_one_axis_is_summed_but_kept(self):
        rows = Tensor(np.ones((4, 3)))
        bias = Tensor(np.ones((1, 3)))
        (rows + bias).sum().backward()
        self.assertEqual(bias.grad.shape, (1, 3))
        np.testing.assert_allclose(bias.grad, [[4.0, 4.0, 4.0]])

    def test_scalar_broadcast_collects_everything(self):
        rows = Tensor(np.ones((2, 3)))
        scale = Tensor(2.0)
        (rows * scale).sum().backward()
        self.assertEqual(scale.grad.shape, ())
        self.assertAlmostEqual(float(scale.grad), 6.0)

    def test_forgetting_to_unbroadcast_is_caught(self):
        # Proof the check bites: a broken add that skips unbroadcast
        # produces a (2,3) gradient for a (3,) input.
        class Broken(Tensor):
            def __add__(self, other):
                other = Tensor._wrap(other)
                out = Tensor(self.data + other.data, (self, other), "+")

                def _backward():
                    self.grad = self.grad + out.grad      # no unbroadcast
                    other.grad = other.grad + out.grad

                out._backward = _backward
                return out

        rows = Broken(np.ones((2, 3)))
        bias = Broken([1.0, 2.0, 3.0])
        (rows + bias).sum().backward()
        self.assertNotEqual(bias.grad.shape, (3,))


class TestSumAndBroadcastAreAdjoint(unittest.TestCase):
    def test_sum_gradient_is_all_ones(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        t.sum().backward()
        np.testing.assert_allclose(t.grad, np.ones((2, 3)))

    def test_axis_sum_gradient_broadcasts_back(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        t.sum(axis=0).sum().backward()
        np.testing.assert_allclose(t.grad, np.ones((2, 3)))

    def test_mean_gradient_is_one_over_n(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0])
        t.mean().backward()
        np.testing.assert_allclose(t.grad, np.full(4, 0.25))

    def test_reshape_gradient_returns_to_the_original_shape(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        t.reshape(6).sum().backward()
        self.assertEqual(t.grad.shape, (2, 3))

    def test_transpose_gradient_transposes_back(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        (t.T * 2.0).sum().backward()
        self.assertEqual(t.grad.shape, (2, 3))
        np.testing.assert_allclose(t.grad, np.full((2, 3), 2.0))


class TestBackwardContract(unittest.TestCase):
    def test_non_scalar_output_rejected(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0]).backward()

    def test_a_size_one_array_counts_as_scalar(self):
        Tensor([[3.0]]).backward()  # should not raise

    def test_reused_tensor_accumulates(self):
        t = Tensor([2.0, 3.0])
        (t * t).sum().backward()
        np.testing.assert_allclose(t.grad, [4.0, 6.0])  # 2x

    def test_zero_grad_clears_everything(self):
        a = Tensor([1.0, 2.0])
        out = (a * a).sum()
        out.backward()
        out.zero_grad()
        self.assertEqual(float(a.grad.sum()), 0.0)

    def test_topological_order_visits_each_node_once(self):
        a = Tensor([1.0])
        shared = a * 3.0
        out = (shared + shared).sum()
        self.assertEqual(sum(1 for n in out.topological_order() if n is shared), 1)


class TestGradientChecks(unittest.TestCase):
    """Element-by-element, against finite differences."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_broadcast_add(self):
        check_gradient(lambda a, b: (a + b).sum(),
                       [self.rng.normal(size=(2, 3)), self.rng.normal(size=(3,))])

    def test_broadcast_multiply(self):
        check_gradient(lambda a, b: (a * b).sum(),
                       [self.rng.normal(size=(2, 3)), self.rng.normal(size=(1, 3))])

    def test_scalar_broadcast(self):
        check_gradient(lambda a, b: (a * b).sum(),
                       [self.rng.normal(size=(2, 3)), np.array(1.7)])

    def test_activations(self):
        for function in (lambda a: a.tanh().sum(),
                         lambda a: a.relu().sum(),
                         lambda a: a.exp().sum()):
            check_gradient(function, [self.rng.normal(size=(3, 2))])

    def test_exp_then_log(self):
        check_gradient(lambda a: (a.exp() + 1.0).log().sum(), [self.rng.normal(size=(4,))])

    def test_mean(self):
        check_gradient(lambda a: a.mean(), [self.rng.normal(size=(2, 3))])

    def test_axis_sum(self):
        check_gradient(lambda a: (a.sum(axis=0) ** 2).sum(), [self.rng.normal(size=(2, 3))])

    def test_reshape(self):
        check_gradient(lambda a: (a.reshape(3, 2) ** 2).sum(), [self.rng.normal(size=(2, 3))])

    def test_transpose(self):
        check_gradient(lambda a: (a.T * 2.0).sum(), [self.rng.normal(size=(2, 3))])

    def test_input_used_twice(self):
        check_gradient(lambda a: (a * a + a).sum(), [self.rng.normal(size=(3,))])

    def test_a_whole_layer(self):
        check_gradient(
            lambda x, w, b: ((x * w).sum(axis=1) + b).tanh().sum(),
            [self.rng.normal(size=(4, 3)), self.rng.normal(size=(3,)), self.rng.normal(size=(4,))],
        )

    def test_shape_mismatch_is_reported(self):
        with self.assertRaises(TypeError):
            check_gradient(lambda a: 5.0, [np.ones(3)])


class TestNumericalGradientCost(unittest.TestCase):
    """Why reverse mode exists."""

    def test_it_needs_two_passes_per_element(self):
        calls = {"count": 0}

        def counted(a):
            calls["count"] += 1
            return (a * a).sum()

        numerical_gradient(counted, [np.ones(10)])
        self.assertEqual(calls["count"], 20)

    def test_backprop_needs_one(self):
        calls = {"count": 0}
        t = Tensor(np.ones(10))

        def counted(a):
            calls["count"] += 1
            return (a * a).sum()

        counted(t).backward()
        self.assertEqual(calls["count"], 1)

    def test_bad_h_rejected(self):
        with self.assertRaises(ValueError):
            numerical_gradient(lambda a: a.sum(), [np.ones(2)], h=0)


if __name__ == "__main__":
    unittest.main()
