"""Day 2 tests — the new derivatives, and the step-size trade-off.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from operations import (
    Value,
    check_gradient,
    numerical_gradient,
    relative_error,
    step_size_sweep,
)


class TestForwardValues(unittest.TestCase):
    def test_exp(self):
        self.assertAlmostEqual(Value(1.0).exp().data, math.e)

    def test_log_inverts_exp(self):
        self.assertAlmostEqual(Value(3.0).exp().log().data, 3.0)

    def test_tanh_is_odd(self):
        self.assertAlmostEqual(Value(1.3).tanh().data, -Value(-1.3).tanh().data)

    def test_tanh_range(self):
        for x in (-50.0, -1.0, 0.0, 1.0, 50.0):
            self.assertGreaterEqual(Value(x).tanh().data, -1.0)
            self.assertLessEqual(Value(x).tanh().data, 1.0)

    def test_sigmoid_at_zero(self):
        self.assertEqual(Value(0.0).sigmoid().data, 0.5)

    def test_sigmoid_range(self):
        for x in (-800.0, -1.0, 0.0, 1.0, 800.0):
            value = Value(x).sigmoid().data
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_sigmoid_does_not_overflow_on_large_negative(self):
        # 1 / (1 + exp(-x)) would raise OverflowError at x = -800.
        self.assertTrue(math.isfinite(Value(-800.0).sigmoid().data))

    def test_sigmoid_symmetry(self):
        self.assertAlmostEqual(Value(2.0).sigmoid().data + Value(-2.0).sigmoid().data, 1.0)

    def test_relu(self):
        self.assertEqual(Value(-2.0).relu().data, 0.0)
        self.assertEqual(Value(2.0).relu().data, 2.0)
        self.assertEqual(Value(0.0).relu().data, 0.0)

    def test_log_of_non_positive_raises(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                Value(bad).log()


class TestNewDerivatives(unittest.TestCase):
    """Every one against finite differences."""

    def test_exp(self):
        check_gradient(lambda a: a.exp(), [0.7])

    def test_log(self):
        check_gradient(lambda a: a.log(), [2.5])

    def test_tanh(self):
        check_gradient(lambda a: a.tanh(), [0.4])

    def test_sigmoid(self):
        check_gradient(lambda a: a.sigmoid(), [-1.2])

    def test_relu_away_from_the_kink(self):
        check_gradient(lambda a: a.relu(), [1.3])
        check_gradient(lambda a: a.relu(), [-1.3])

    def test_a_neuron(self):
        check_gradient(lambda w, x, b: (w * x + b).tanh(), [0.5, 2.0, -1.0])

    def test_log_sum_exp(self):
        check_gradient(lambda a, b: (a.exp() + b.exp()).log(), [1.0, 2.0])

    def test_negative_log_sigmoid(self):
        # The shape of binary cross-entropy.
        check_gradient(lambda a: -(a.sigmoid().log()), [0.3])

    def test_composition_of_everything(self):
        check_gradient(
            lambda a, b: ((a * b).tanh() + a.exp()).log() * b.sigmoid(),
            [0.4, 0.9],
        )


class TestDerivativeIdentities(unittest.TestCase):
    """Known closed forms, checked against the implementation."""

    def test_exp_derivative_equals_its_value(self):
        node = Value(1.3)
        out = node.exp()
        out.backward()
        self.assertAlmostEqual(node.grad, out.data)

    def test_tanh_derivative_is_one_minus_square(self):
        node = Value(0.8)
        out = node.tanh()
        out.backward()
        self.assertAlmostEqual(node.grad, 1 - out.data ** 2)

    def test_sigmoid_derivative_is_s_times_one_minus_s(self):
        node = Value(-0.6)
        out = node.sigmoid()
        out.backward()
        self.assertAlmostEqual(node.grad, out.data * (1 - out.data))

    def test_log_derivative_is_reciprocal(self):
        node = Value(4.0)
        node.log().backward()
        self.assertAlmostEqual(node.grad, 0.25)

    def test_sigmoid_derivative_peaks_at_zero(self):
        peak = Value(0.0)
        peak.sigmoid().backward()
        for x in (-2.0, -0.5, 0.5, 2.0):
            other = Value(x)
            other.sigmoid().backward()
            self.assertLess(other.grad, peak.grad)


class TestSaturation(unittest.TestCase):
    """The vanishing gradient, measured."""

    def test_tanh_gradient_collapses_with_magnitude(self):
        gradients = []
        for x in (0.0, 1.0, 3.0, 6.0, 10.0):
            node = Value(x)
            node.tanh().backward()
            gradients.append(node.grad)
        self.assertEqual(gradients, sorted(gradients, reverse=True))
        self.assertLess(gradients[-1], 1e-7)

    def test_a_deep_stack_multiplies_them_together(self):
        # Ten saturating layers, and the input gradient is gone.
        node = Value(3.0)
        out = node
        for _ in range(10):
            out = out.tanh() * 3.0
        out.backward()
        self.assertLess(abs(node.grad), 1e-2)

    def test_relu_does_not_saturate_on_the_positive_side(self):
        node = Value(50.0)
        node.relu().backward()
        self.assertEqual(node.grad, 1.0)


class TestReluAtZero(unittest.TestCase):
    """Not differentiable there, and the code says so rather than hiding it."""

    def test_autograd_picks_zero(self):
        node = Value(0.0)
        node.relu().backward()
        self.assertEqual(node.grad, 0.0)

    def test_the_finite_difference_disagrees(self):
        # It straddles the kink and reports the average of the two sides.
        self.assertAlmostEqual(numerical_gradient(lambda a: a.relu(), [0.0], 0), 0.5, places=6)

    def test_so_the_gradient_check_would_fail_there(self):
        with self.assertRaises(AssertionError):
            check_gradient(lambda a: a.relu(), [0.0])


class TestRelativeError(unittest.TestCase):
    def test_identical_values(self):
        self.assertEqual(relative_error(1.0, 1.0), 0.0)

    def test_scale_free(self):
        # The same proportional disagreement at two very different scales.
        small = relative_error(1e-6, 2e-6)
        large = relative_error(1e6, 2e6)
        self.assertAlmostEqual(large, 0.5)
        self.assertLess(small, large)  # max(1, ...) floors the tiny case

    def test_near_zero_stays_finite(self):
        self.assertTrue(math.isfinite(relative_error(0.0, 0.0)))


class TestStepSizeSweep(unittest.TestCase):
    """The U-curve: truncation on one side, cancellation on the other."""

    def setUp(self):
        self.sweep = step_size_sweep(lambda a: (a * a * a).exp(), [0.6])
        self.errors = [error for _, error in self.sweep]

    def test_error_is_not_monotonic(self):
        # If it were, "smaller h is better" would be true, and it is not.
        self.assertNotEqual(self.errors, sorted(self.errors))
        self.assertNotEqual(self.errors, sorted(self.errors, reverse=True))

    def test_the_best_step_is_in_the_middle(self):
        best = min(range(len(self.errors)), key=lambda i: self.errors[i])
        self.assertNotIn(best, (0, len(self.errors) - 1))

    def test_large_h_is_dominated_by_truncation(self):
        # Error falls roughly as h^2: shrinking h by 10 cuts it by ~100.
        (h1, e1), (h2, e2) = self.sweep[0], self.sweep[1]
        self.assertAlmostEqual(h1 / h2, 10.0)
        self.assertGreater(e1 / e2, 50.0)

    def test_tiny_h_is_dominated_by_cancellation(self):
        self.assertGreater(self.errors[-1], self.errors[-3])

    def test_the_conventional_default_is_close_to_optimal(self):
        best = min(self.errors)
        at_default = dict(self.sweep)[1e-5]
        self.assertLess(at_default, 1e4 * best)


class TestGradientCheckTolerance(unittest.TestCase):
    def test_a_bad_step_size_can_fail_a_correct_gradient(self):
        # The gradient is right; the check is wrong. Worth knowing before
        # spending an afternoon on a derivative that was fine.
        check_gradient(lambda a: (a * a * a).exp(), [0.6], h=1e-5)
        with self.assertRaises(AssertionError):
            check_gradient(lambda a: (a * a * a).exp(), [0.6], h=1e-14)

    def test_non_value_return_rejected(self):
        with self.assertRaises(TypeError):
            check_gradient(lambda a: 5.0, [1.0])

    def test_bad_index_rejected(self):
        with self.assertRaises(IndexError):
            numerical_gradient(lambda a: a.exp(), [1.0], 3)

    def test_non_positive_h_rejected(self):
        with self.assertRaises(ValueError):
            numerical_gradient(lambda a: a.exp(), [1.0], 0, h=-1e-5)


if __name__ == "__main__":
    unittest.main()
