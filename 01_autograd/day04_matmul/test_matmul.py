"""Day 4 tests — matmul forward, backward, batching, and shape discipline.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from matmul import Tensor, check_gradient, linear, numerical_gradient


class TestMatmulForward(unittest.TestCase):
    def test_identity(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        identity = Tensor(np.eye(2))
        np.testing.assert_allclose((a @ identity).data, a.data)

    def test_shape(self):
        self.assertEqual((Tensor(np.ones((4, 3))) @ Tensor(np.ones((3, 5)))).shape, (4, 5))

    def test_values_match_numpy(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
        np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)

    def test_batched_shape(self):
        self.assertEqual((Tensor(np.ones((2, 3, 4))) @ Tensor(np.ones((2, 4, 5)))).shape, (2, 3, 5))

    def test_batch_broadcast_shape(self):
        self.assertEqual((Tensor(np.ones((2, 3, 4))) @ Tensor(np.ones((4, 5)))).shape, (2, 3, 5))

    def test_rmatmul_with_a_raw_array(self):
        out = np.ones((2, 3)) @ Tensor(np.ones((3, 4)))
        self.assertEqual(out.shape, (2, 4))


class TestMatmulShapeDiscipline(unittest.TestCase):
    def test_one_dimensional_operands_rejected(self):
        # NumPy would silently promote and then demote, changing what the
        # gradient shapes mean.
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0]) @ Tensor([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            Tensor([[1.0, 2.0]]) @ Tensor([1.0, 2.0])

    def test_mismatched_inner_dimension_named(self):
        with self.assertRaises(ValueError) as caught:
            Tensor(np.ones((3, 4))) @ Tensor(np.ones((5, 2)))
        self.assertIn("4 != 5", str(caught.exception))

    def test_a_vector_works_once_reshaped(self):
        vector = Tensor([1.0, 2.0]).reshape(1, 2)
        self.assertEqual((vector @ Tensor(np.ones((2, 3)))).shape, (1, 3))


class TestMatmulBackward(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_gradient_shapes_match_the_inputs(self):
        a = Tensor(self.rng.normal(size=(4, 3)))
        b = Tensor(self.rng.normal(size=(3, 5)))
        (a @ b).sum().backward()
        self.assertEqual(a.grad.shape, (4, 3))
        self.assertEqual(b.grad.shape, (3, 5))

    def test_closed_form_da(self):
        # dL/dA = dL/dC @ B^T, and with L = sum(C) the incoming gradient
        # is all ones, so dL/dA is B summed along its output axis.
        a = Tensor(self.rng.normal(size=(2, 3)))
        b = Tensor(self.rng.normal(size=(3, 4)))
        (a @ b).sum().backward()
        np.testing.assert_allclose(a.grad, np.ones((2, 4)) @ b.data.T)

    def test_closed_form_db(self):
        a = Tensor(self.rng.normal(size=(2, 3)))
        b = Tensor(self.rng.normal(size=(3, 4)))
        (a @ b).sum().backward()
        np.testing.assert_allclose(b.grad, a.data.T @ np.ones((2, 4)))

    def test_identity_gradient(self):
        a = Tensor(self.rng.normal(size=(3, 3)))
        identity = Tensor(np.eye(3))
        (a @ identity).sum().backward()
        np.testing.assert_allclose(a.grad, np.ones((3, 3)))

    def test_a_times_its_own_transpose_collects_both_paths(self):
        a = Tensor(self.rng.normal(size=(3, 4)))
        (a @ a.T).sum().backward()
        numeric = numerical_gradient(lambda p: (p @ p.T).sum(), [a.data])[0]
        np.testing.assert_allclose(a.grad, numeric, atol=1e-6)

    def test_batched_gradient_shapes(self):
        a = Tensor(self.rng.normal(size=(2, 3, 4)))
        b = Tensor(self.rng.normal(size=(2, 4, 5)))
        (a @ b).sum().backward()
        self.assertEqual(a.grad.shape, (2, 3, 4))
        self.assertEqual(b.grad.shape, (2, 4, 5))

    def test_broadcast_batch_gradient_is_summed_over_the_batch(self):
        a = Tensor(self.rng.normal(size=(2, 3, 4)))
        b = Tensor(self.rng.normal(size=(4, 5)))
        (a @ b).sum().backward()
        self.assertEqual(b.grad.shape, (4, 5))


class TestGradientChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_plain_matmul(self):
        check_gradient(lambda p, q: (p @ q).sum(),
                       [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4, 2))])

    def test_linear_layer(self):
        check_gradient(lambda p, q, r: linear(p, q, r).sum(),
                       [self.rng.normal(size=(5, 3)), self.rng.normal(size=(3, 2)),
                        self.rng.normal(size=(2,))])

    def test_two_layers_with_an_activation(self):
        check_gradient(lambda p, q, r: ((p @ q).tanh() @ r).sum(),
                       [self.rng.normal(size=(4, 3)), self.rng.normal(size=(3, 3)),
                        self.rng.normal(size=(3, 2))])

    def test_matmul_with_transpose(self):
        check_gradient(lambda p: (p @ p.T).sum(), [self.rng.normal(size=(3, 4))])

    def test_batched(self):
        check_gradient(lambda p, q: (p @ q).sum(),
                       [self.rng.normal(size=(2, 3, 4)), self.rng.normal(size=(2, 4, 3))])

    def test_batch_broadcast(self):
        check_gradient(lambda p, q: (p @ q).sum(),
                       [self.rng.normal(size=(2, 3, 4)), self.rng.normal(size=(4, 3))])

    def test_mean_squared_error(self):
        check_gradient(lambda p, q, t: (((p @ q) - t) ** 2).mean(),
                       [self.rng.normal(size=(6, 3)), self.rng.normal(size=(3, 2)),
                        self.rng.normal(size=(6, 2))])

    def test_reshape_between_matmuls(self):
        check_gradient(lambda p, q: ((p @ q).reshape(1, 6) ** 2).sum(),
                       [self.rng.normal(size=(2, 3)), self.rng.normal(size=(3, 3))])

    def test_a_wrong_matmul_backward_is_caught(self):
        # Swap the two transposes - a plausible typo that produces
        # correctly *shaped* gradients only for square matrices.
        class Broken(Tensor):
            def __matmul__(self, other):
                other = Tensor._wrap(other)
                out = Tensor(self.data @ other.data, (self, other), "@")

                def _backward():
                    # dA should use B^T; this uses B.
                    self.grad = self.grad + out.grad @ other.data
                    self.grad = self.grad * 1.0
                    other.grad = other.grad + self.data.T @ out.grad

                out._backward = _backward
                return out

        a = Broken(self.rng.normal(size=(3, 3)))
        b = Broken(self.rng.normal(size=(3, 3)))
        (a @ b).sum().backward()
        numeric = numerical_gradient(lambda p, q: (p @ q).sum(), [a.data, b.data])[0]
        self.assertGreater(float(np.abs(a.grad - numeric).max()), 1e-3)


class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_output_shape(self):
        out = linear(Tensor(self.rng.normal(size=(8, 4))),
                     Tensor(self.rng.normal(size=(4, 3))),
                     Tensor(np.zeros(3)))
        self.assertEqual(out.shape, (8, 3))

    def test_bias_gradient_sums_over_the_batch(self):
        x = Tensor(self.rng.normal(size=(8, 4)))
        weight = Tensor(self.rng.normal(size=(4, 3)))
        bias = Tensor(np.zeros(3))
        linear(x, weight, bias).sum().backward()
        # Every one of the 8 rows used the same bias.
        np.testing.assert_allclose(bias.grad, np.full(3, 8.0))

    def test_zero_weights_give_the_bias(self):
        out = linear(Tensor(self.rng.normal(size=(4, 3))),
                     Tensor(np.zeros((3, 2))),
                     Tensor([1.0, -1.0]))
        np.testing.assert_allclose(out.data, np.tile([1.0, -1.0], (4, 1)))

    def test_one_gradient_step_reduces_a_squared_error(self):
        # The premise of training, now with matrices.
        x = self.rng.normal(size=(16, 3))
        target = self.rng.normal(size=(16, 2))
        weight = Tensor(self.rng.normal(size=(3, 2)) * 0.1)
        bias = Tensor(np.zeros(2))

        def loss_of(w, b):
            return float((((Tensor(x) @ w + b) - Tensor(target)) ** 2).mean().data)

        before = loss_of(weight, bias)
        loss = ((linear(Tensor(x), weight, bias) - Tensor(target)) ** 2).mean()
        loss.backward()
        stepped_w = Tensor(weight.data - 0.05 * weight.grad)
        stepped_b = Tensor(bias.data - 0.05 * bias.grad)
        self.assertLess(loss_of(stepped_w, stepped_b), before)

    def test_repeated_steps_converge(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(32, 3))
        true_w = rng.normal(size=(3, 1))
        target = x @ true_w
        weight = Tensor(np.zeros((3, 1)))
        losses = []
        for _ in range(60):
            loss = (((Tensor(x) @ weight) - Tensor(target)) ** 2).mean()
            loss.backward()
            losses.append(float(loss.data))
            weight = Tensor(weight.data - 0.1 * weight.grad)
        self.assertLess(losses[-1], losses[0] / 100)
        np.testing.assert_allclose(weight.data, true_w, atol=0.05)


if __name__ == "__main__":
    unittest.main()
