"""Day 6 tests — the two overflows, and the gradient that cancels.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from losses import (
    Tensor,
    accuracy,
    check_gradient,
    cross_entropy,
    cross_entropy_gradient,
    log_softmax,
    naive_softmax,
    one_hot,
    softmax,
)


class TestSoftmaxValues(unittest.TestCase):
    def test_uniform_input_gives_uniform_output(self):
        np.testing.assert_allclose(softmax(Tensor([1.0, 1.0, 1.0])).data, [1 / 3] * 3)

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        for scale in (0.01, 1.0, 100.0, 1000.0):
            rows = softmax(Tensor(rng.normal(size=(4, 6)) * scale)).data
            np.testing.assert_allclose(rows.sum(axis=-1), np.ones(4))

    def test_output_is_in_the_unit_interval(self):
        rng = np.random.default_rng(1)
        values = softmax(Tensor(rng.normal(size=(5, 5)) * 50)).data
        self.assertGreaterEqual(values.min(), 0.0)
        self.assertLessEqual(values.max(), 1.0)

    def test_order_is_preserved(self):
        logits = np.array([3.0, 1.0, 2.0])
        self.assertEqual(list(np.argsort(softmax(Tensor(logits)).data)),
                         list(np.argsort(logits)))

    def test_matches_the_definition_where_the_definition_works(self):
        logits = np.array([[1.0, 2.0, 0.5]])
        np.testing.assert_allclose(softmax(Tensor(logits)).data, naive_softmax(logits))

    def test_two_classes_reduce_to_the_sigmoid(self):
        # softmax([x, 0])[0] == sigmoid(x)
        for x in (-2.0, 0.0, 3.0):
            expected = 1.0 / (1.0 + math.exp(-x))
            self.assertAlmostEqual(float(softmax(Tensor([x, 0.0])).data[0]), expected)


class TestOverflow(unittest.TestCase):
    """The failure the max-subtraction exists to prevent."""

    def test_the_naive_form_produces_nan(self):
        with np.errstate(over="ignore", invalid="ignore"):
            result = naive_softmax(np.array([1000.0, 0.0, 1.0]))
        self.assertFalse(np.all(np.isfinite(result)))

    def test_the_stable_form_does_not(self):
        result = softmax(Tensor([1000.0, 0.0, 1.0])).data
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_it_starts_failing_around_710(self):
        # exp(709) is finite in float64; exp(710) is not.
        with np.errstate(over="ignore", invalid="ignore"):
            self.assertTrue(np.all(np.isfinite(naive_softmax(np.array([700.0, 0.0])))))
            self.assertFalse(np.all(np.isfinite(naive_softmax(np.array([800.0, 0.0])))))

    def test_shift_invariance_is_exact(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=6) * 3
        for shift in (-100.0, -1.0, 0.0, 50.0):
            np.testing.assert_allclose(softmax(Tensor(logits)).data,
                                       softmax(Tensor(logits + shift)).data)

    def test_very_negative_inputs_stay_finite(self):
        result = softmax(Tensor([-1000.0, -1000.0])).data
        np.testing.assert_allclose(result, [0.5, 0.5])


class TestLogSoftmax(unittest.TestCase):
    def test_two_equal_logits(self):
        np.testing.assert_allclose(log_softmax(Tensor([0.0, 0.0])).data,
                                   [-math.log(2), -math.log(2)])

    def test_agrees_with_log_of_softmax_in_the_safe_range(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(3, 4))
        np.testing.assert_allclose(log_softmax(Tensor(logits)).data,
                                   np.log(softmax(Tensor(logits)).data), atol=1e-12)

    def test_survives_where_the_two_step_form_does_not(self):
        confident = np.array([[0.0, -800.0]])
        with np.errstate(divide="ignore"):
            two_step = np.log(softmax(Tensor(confident)).data)
        fused = log_softmax(Tensor(confident)).data
        self.assertFalse(np.all(np.isfinite(two_step)))
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertAlmostEqual(float(fused[0, 1]), -800.0, places=6)

    def test_exponentiates_back_to_softmax(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(3, 5))
        np.testing.assert_allclose(np.exp(log_softmax(Tensor(logits)).data),
                                   softmax(Tensor(logits)).data)

    def test_every_value_is_negative(self):
        rng = np.random.default_rng(3)
        self.assertLessEqual(log_softmax(Tensor(rng.normal(size=(4, 4)))).data.max(), 0.0)


class TestOneHot(unittest.TestCase):
    def test_encoding(self):
        np.testing.assert_array_equal(one_hot([0, 2], 3),
                                      [[1, 0, 0], [0, 0, 1]])

    def test_rows_sum_to_one(self):
        self.assertTrue(np.all(one_hot([0, 1, 2, 1], 3).sum(axis=1) == 1))

    def test_out_of_range_rejected(self):
        with self.assertRaises(IndexError):
            one_hot([3], 3)

    def test_negative_rejected(self):
        with self.assertRaises(IndexError):
            one_hot([-1], 3)

    def test_float_targets_rejected(self):
        with self.assertRaises(TypeError):
            one_hot(np.array([0.0, 1.0]), 2)

    def test_two_dimensional_targets_rejected(self):
        with self.assertRaises(ValueError):
            one_hot(np.array([[0, 1]]), 2)


class TestCrossEntropy(unittest.TestCase):
    def test_uniform_over_k_classes_is_log_k(self):
        for classes in (2, 5, 10, 100):
            loss = cross_entropy(Tensor([[0.0] * classes]), [0])
            self.assertAlmostEqual(float(loss.data), math.log(classes), places=9)

    def test_confident_and_right_is_near_zero(self):
        self.assertLess(float(cross_entropy(Tensor([[20.0, 0.0]]), [0]).data), 1e-8)

    def test_confident_and_wrong_is_large(self):
        self.assertAlmostEqual(float(cross_entropy(Tensor([[10.0, 0.0]]), [1]).data),
                               10.0, places=4)

    def test_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            logits = Tensor(rng.normal(size=(4, 5)) * 5)
            self.assertGreaterEqual(float(cross_entropy(logits, [0, 1, 2, 3]).data), 0.0)

    def test_mean_and_sum_reductions(self):
        logits = Tensor(np.zeros((4, 2)))
        mean = float(cross_entropy(logits, [0, 0, 0, 0], "mean").data)
        total = float(cross_entropy(logits, [0, 0, 0, 0], "sum").data)
        self.assertAlmostEqual(total, 4 * mean)

    def test_survives_a_confident_wrong_answer(self):
        # The case that makes log(softmax(x)) produce inf.
        loss = cross_entropy(Tensor([[0.0, -800.0]]), [1])
        self.assertTrue(math.isfinite(float(loss.data)))
        self.assertAlmostEqual(float(loss.data), 800.0, places=4)

    def test_unknown_reduction_rejected(self):
        with self.assertRaises(ValueError):
            cross_entropy(Tensor([[0.0, 0.0]]), [0], "median")

    def test_non_2d_logits_rejected(self):
        with self.assertRaises(ValueError):
            cross_entropy(Tensor([0.0, 0.0]), [0])


class TestTheFusedGradient(unittest.TestCase):
    """The cancellation that makes softmax+CE cheap."""

    def test_autograd_matches_p_minus_y_over_n(self):
        rng = np.random.default_rng(0)
        logits = Tensor(rng.normal(size=(4, 3)))
        targets = [0, 2, 1, 2]
        cross_entropy(logits, targets).backward()
        np.testing.assert_allclose(logits.grad,
                                   cross_entropy_gradient(logits.data, targets), atol=1e-12)

    def test_the_gradient_sums_to_zero_per_row(self):
        # p sums to 1 and y sums to 1, so their difference sums to 0 -
        # a shift in all logits cannot change the loss.
        rng = np.random.default_rng(1)
        logits = Tensor(rng.normal(size=(5, 4)))
        cross_entropy(logits, [0, 1, 2, 3, 0]).backward()
        np.testing.assert_allclose(logits.grad.sum(axis=1), np.zeros(5), atol=1e-12)

    def test_the_correct_class_gets_a_negative_gradient(self):
        logits = Tensor([[0.0, 0.0, 0.0]])
        cross_entropy(logits, [1]).backward()
        self.assertLess(logits.grad[0, 1], 0.0)
        self.assertGreater(logits.grad[0, 0], 0.0)

    def test_a_perfect_prediction_has_almost_no_gradient(self):
        logits = Tensor([[30.0, 0.0]])
        cross_entropy(logits, [0]).backward()
        self.assertLess(float(np.abs(logits.grad).max()), 1e-10)


class TestGradientChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_softmax(self):
        check_gradient(lambda z: softmax(z).sum(), [self.rng.normal(size=(3, 4))])

    def test_softmax_squared(self):
        check_gradient(lambda z: (softmax(z) ** 2).sum(), [self.rng.normal(size=(3, 4))])

    def test_log_softmax(self):
        check_gradient(lambda z: (log_softmax(z) ** 2).sum(), [self.rng.normal(size=(3, 4))])

    def test_cross_entropy(self):
        check_gradient(lambda z: cross_entropy(z, [0, 2, 1]), [self.rng.normal(size=(3, 4))])

    def test_linear_then_cross_entropy(self):
        check_gradient(lambda x, w: cross_entropy(x @ w, [1, 0]),
                       [self.rng.normal(size=(2, 5)), self.rng.normal(size=(5, 3))])

    def test_gradient_check_holds_at_large_logits(self):
        check_gradient(lambda z: cross_entropy(z, [0, 1]),
                       [self.rng.normal(size=(2, 3)) * 20])


class TestAccuracy(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(accuracy(Tensor([[5.0, 0.0], [0.0, 5.0]]), [0, 1]), 1.0)

    def test_all_wrong(self):
        self.assertEqual(accuracy(Tensor([[5.0, 0.0], [0.0, 5.0]]), [1, 0]), 0.0)

    def test_half(self):
        self.assertEqual(accuracy(Tensor([[5.0, 0.0], [0.0, 5.0]]), [0, 0]), 0.5)

    def test_is_unaffected_by_a_shift(self):
        logits = Tensor([[1.0, 2.0], [3.0, 0.0]])
        shifted = Tensor(logits.data + 100.0)
        self.assertEqual(accuracy(logits, [1, 0]), accuracy(shifted, [1, 0]))


if __name__ == "__main__":
    unittest.main()
