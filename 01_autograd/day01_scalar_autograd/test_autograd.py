"""Day 1 tests — the graph, the chain rule, and the finite-difference check.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from autograd import Value, check_gradient, numerical_gradient


class TestValue(unittest.TestCase):
    def test_stores_data_as_float(self):
        self.assertEqual(Value(3).data, 3.0)
        self.assertIsInstance(Value(3).data, float)

    def test_grad_starts_at_zero(self):
        self.assertEqual(Value(1.0).grad, 0.0)

    def test_repr_shows_data_and_grad(self):
        self.assertIn("data=2", repr(Value(2.0)))

    def test_rejects_non_numbers(self):
        for bad in ("1.0", None, [1.0]):
            with self.assertRaises(TypeError):
                Value(bad)

    def test_rejects_bool(self):
        # True is an int in Python and would become Value(1.0) silently.
        with self.assertRaises(TypeError):
            Value(True)


class TestForwardArithmetic(unittest.TestCase):
    def test_add(self):
        self.assertEqual((Value(2.0) + Value(3.0)).data, 5.0)

    def test_multiply(self):
        self.assertEqual((Value(2.0) * Value(3.0)).data, 6.0)

    def test_power(self):
        self.assertEqual((Value(2.0) ** 3).data, 8.0)

    def test_negate_subtract_divide(self):
        self.assertEqual((-Value(2.0)).data, -2.0)
        self.assertEqual((Value(5.0) - Value(3.0)).data, 2.0)
        self.assertEqual((Value(6.0) / Value(3.0)).data, 2.0)

    def test_constants_are_wrapped(self):
        self.assertEqual((Value(2.0) + 3).data, 5.0)
        self.assertEqual((Value(2.0) * 3).data, 6.0)

    def test_reflected_operators(self):
        self.assertEqual((3 + Value(2.0)).data, 5.0)
        self.assertEqual((3 * Value(2.0)).data, 6.0)
        self.assertEqual((3 - Value(2.0)).data, 1.0)
        self.assertEqual((6 / Value(3.0)).data, 2.0)

    def test_value_exponent_rejected(self):
        # d(a**b)/db needs a**b * ln(a), which is not implemented.
        with self.assertRaises(TypeError):
            Value(2.0) ** Value(3.0)

    def test_non_numeric_exponent_rejected(self):
        with self.assertRaises(TypeError):
            Value(2.0) ** "3"


class TestBackward(unittest.TestCase):
    def test_product_rule(self):
        a, b = Value(2.0), Value(3.0)
        (a * b).backward()
        self.assertEqual((a.grad, b.grad), (3.0, 2.0))

    def test_addition_passes_gradient_through(self):
        a, b = Value(2.0), Value(3.0)
        (a + b).backward()
        self.assertEqual((a.grad, b.grad), (1.0, 1.0))

    def test_power_rule(self):
        a = Value(2.0)
        (a ** 3).backward()
        self.assertAlmostEqual(a.grad, 12.0)  # 3 * 2^2

    def test_output_gradient_is_one(self):
        a = Value(2.0)
        out = a * 3
        out.backward()
        self.assertEqual(out.grad, 1.0)

    def test_reused_value_accumulates(self):
        # The += in _backward. With = the answer would be 1.0.
        x = Value(3.0)
        (x + x).backward()
        self.assertEqual(x.grad, 2.0)

    def test_reused_value_in_a_product(self):
        x = Value(3.0)
        (x * x).backward()
        self.assertEqual(x.grad, 6.0)  # 2x

    def test_deep_chain(self):
        x = Value(2.0)
        out = x
        for _ in range(5):
            out = out * x
        out.backward()
        self.assertAlmostEqual(out.data, 64.0)   # x^6
        self.assertAlmostEqual(x.grad, 6 * 2.0 ** 5)

    def test_backward_accumulates_across_calls(self):
        # Documented behaviour, not a bug: it is what makes gradient
        # accumulation over mini-batches work.
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        first = a.grad
        out._backward()
        self.assertEqual(a.grad, 2 * first)

    def test_zero_grad_clears_the_graph(self):
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.backward()
        out.zero_grad()
        self.assertEqual((a.grad, b.grad, out.grad), (0.0, 0.0, 0.0))


class TestTopologicalOrder(unittest.TestCase):
    def test_inputs_come_before_outputs(self):
        a = Value(2.0)
        out = a * a + a
        ordered = out.topological_order()
        self.assertIs(ordered[-1], out)
        self.assertLess(ordered.index(a), ordered.index(out))

    def test_each_node_appears_once(self):
        a = Value(2.0)
        out = a * a + a * a
        ordered = out.topological_order()
        self.assertEqual(len(ordered), len({id(n) for n in ordered}))

    def test_a_leaf_is_its_own_order(self):
        a = Value(2.0)
        self.assertEqual(a.topological_order(), [a])

    def test_shared_subexpression_is_visited_once(self):
        a = Value(2.0)
        shared = a * 3
        out = shared + shared
        self.assertEqual(sum(1 for n in out.topological_order() if n is shared), 1)


class TestNumericalGradient(unittest.TestCase):
    def test_matches_a_known_derivative(self):
        self.assertAlmostEqual(
            numerical_gradient(lambda a, b: a * b, [2.0, 3.0], 0), 3.0, places=6
        )

    def test_central_difference_is_symmetric(self):
        # f(x) = x^2 has derivative 2x; at x=0 the central difference is
        # exactly 0 while a forward difference would report h.
        self.assertAlmostEqual(numerical_gradient(lambda a: a * a, [0.0], 0), 0.0, places=9)

    def test_index_out_of_range_rejected(self):
        with self.assertRaises(IndexError):
            numerical_gradient(lambda a: a, [1.0], 5)

    def test_non_positive_h_rejected(self):
        with self.assertRaises(ValueError):
            numerical_gradient(lambda a: a, [1.0], 0, h=0)


class TestGradientCheck(unittest.TestCase):
    """The habit this whole repo is built on."""

    def test_polynomial(self):
        check_gradient(lambda a: a * a * a + 2 * a, [1.5])

    def test_product_and_sum(self):
        self.assertEqual(check_gradient(lambda a, b: a * b + a, [2.0, 3.0]), [4.0, 2.0])

    def test_division(self):
        check_gradient(lambda a, b: a / b, [6.0, 3.0])

    def test_difference_of_squares(self):
        check_gradient(lambda a, b: (a + b) * (a - b), [5.0, 2.0])

    def test_reused_inputs(self):
        check_gradient(lambda a, b: a * a + a * b + b * b, [1.5, -2.5])

    def test_negative_powers(self):
        check_gradient(lambda a: a ** -2, [3.0])

    def test_fractional_powers(self):
        check_gradient(lambda a: a ** 0.5, [4.0])

    def test_deeply_nested(self):
        check_gradient(lambda a, b: ((a * b + a) * (b - a)) / (a + 1.0), [2.0, 3.0])

    def test_non_value_return_rejected(self):
        with self.assertRaises(TypeError):
            check_gradient(lambda a: 5.0, [1.0])

    def test_a_wrong_gradient_is_caught(self):
        # Proof the check can fail: break the product rule on purpose.
        class Broken(Value):
            def __mul__(self, other):
                other = Value._wrap(other)
                out = Value(self.data * other.data, (self, other), "*")

                def _backward():
                    self.grad += out.grad      # missing the * other.data
                    other.grad += self.data * out.grad

                out._backward = _backward
                return out

        values = [Broken(2.0), Broken(3.0)]
        output = values[0] * values[1]
        output.backward()
        numeric = numerical_gradient(lambda a, b: a * b, [2.0, 3.0], 0)
        self.assertNotAlmostEqual(values[0].grad, numeric, places=3)


class TestKnownDerivatives(unittest.TestCase):
    """Spot checks against derivatives worked out by hand."""

    def test_quadratic_minimum_has_zero_gradient(self):
        # f(x) = (x - 3)^2 is flat at x = 3.
        x = Value(3.0)
        ((x - 3.0) ** 2).backward()
        self.assertAlmostEqual(x.grad, 0.0)

    def test_gradient_points_uphill(self):
        # Left of the minimum the slope is negative, right of it positive.
        left, right = Value(1.0), Value(5.0)
        ((left - 3.0) ** 2).backward()
        ((right - 3.0) ** 2).backward()
        self.assertLess(left.grad, 0.0)
        self.assertGreater(right.grad, 0.0)

    def test_one_step_of_gradient_descent_reduces_the_loss(self):
        # The entire premise of training, on the smallest possible model.
        x = Value(5.0)
        loss = (x - 3.0) ** 2
        loss.backward()
        stepped = Value(x.data - 0.1 * x.grad)
        self.assertLess(((stepped - 3.0) ** 2).data, loss.data)

    def test_agrees_with_math_module_on_a_composite(self):
        # d/dx (x^2 + 1)^3 = 6x(x^2 + 1)^2
        x = Value(1.5)
        ((x * x + 1.0) ** 3).backward()
        expected = 6 * 1.5 * (1.5 ** 2 + 1.0) ** 2
        self.assertAlmostEqual(x.grad, expected, places=9)

    def test_no_nan_or_inf_on_ordinary_input(self):
        x, y = Value(2.0), Value(-3.0)
        out = (x * y + x ** 2) / (y - 1.0)
        out.backward()
        for value in (out.data, x.grad, y.grad):
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
