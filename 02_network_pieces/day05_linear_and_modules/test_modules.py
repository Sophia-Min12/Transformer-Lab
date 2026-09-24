"""Day 5 tests — parameter discovery, initialization scale, and the layer.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from modules import (
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Tanh,
    Tensor,
    check_gradient,
    init_scale,
)


class TestParameterDiscovery(unittest.TestCase):
    def test_a_linear_layer_has_weight_and_bias(self):
        self.assertEqual([p.shape for p in Linear(3, 2, seed=0).parameters()], [(3, 2), (2,)])

    def test_bias_can_be_omitted(self):
        self.assertEqual(len(Linear(3, 2, bias=False, seed=0).parameters()), 1)

    def test_nested_modules_are_found(self):
        net = Sequential(Linear(2, 3, seed=0), Tanh(), Linear(3, 2, seed=1))
        self.assertEqual(len(net.parameters()), 4)

    def test_modules_in_a_list_are_found(self):
        stack = ModuleList([Linear(2, 2, seed=0), Linear(2, 2, seed=1)])
        self.assertEqual(len(stack.parameters()), 4)

    def test_attributes_and_lists_together(self):
        class Both(Module):
            def __init__(self):
                self.first = Linear(2, 2, seed=0)
                self.rest = ModuleList([Linear(2, 2, seed=1), Linear(2, 2, seed=2)])

        self.assertEqual(len(Both().parameters()), 6)

    def test_a_shared_parameter_is_listed_once(self):
        # Weight tying (Day 15) reuses one tensor in two places; listing it
        # twice would apply the update twice.
        shared = Linear(2, 2, seed=0)

        class Tied(Module):
            def __init__(self):
                self.a = shared
                self.b = shared

        self.assertEqual(len(Tied().parameters()), 2)

    def test_parameter_count(self):
        self.assertEqual(Linear(8, 16, seed=0).parameter_count(), 8 * 16 + 16)

    def test_zero_grad_clears_everything(self):
        net = Sequential(Linear(3, 2, seed=0))
        (net(Tensor(np.ones((4, 3)))) ** 2).sum().backward()
        self.assertGreater(float(np.abs(net.parameters()[0].grad).sum()), 0.0)
        net.zero_grad()
        for parameter in net.parameters():
            self.assertEqual(float(np.abs(parameter.grad).sum()), 0.0)

    def test_forward_must_be_implemented(self):
        class Empty(Module):
            pass

        with self.assertRaises(NotImplementedError):
            Empty()(Tensor([1.0]))


class TestInitScale(unittest.TestCase):
    def test_xavier(self):
        self.assertAlmostEqual(init_scale(100, 100, "xavier"), math.sqrt(2 / 200))

    def test_he_is_larger_than_lecun(self):
        # ReLU zeroes half its input, so it needs twice the variance.
        self.assertAlmostEqual(init_scale(100, 100, "he") ** 2,
                               2 * init_scale(100, 100, "lecun") ** 2)

    def test_lecun(self):
        self.assertAlmostEqual(init_scale(64, 10, "lecun"), math.sqrt(1 / 64))

    def test_scale_shrinks_as_fan_in_grows(self):
        scales = [init_scale(n, 10, "lecun") for n in (10, 100, 1000)]
        self.assertEqual(scales, sorted(scales, reverse=True))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            init_scale(10, 10, "magic")

    def test_non_positive_fan_rejected(self):
        with self.assertRaises(ValueError):
            init_scale(0, 10)


class TestInitializationKeepsSignalAlive(unittest.TestCase):
    """The measurement: a scale that is wrong kills a deep network."""

    @staticmethod
    def depth_profile(factor: float, depth: int = 30, width: int = 64) -> list[float]:
        x = Tensor(np.random.default_rng(0).normal(size=(32, width)))
        stds = []
        for layer_index in range(depth):
            scale = init_scale(width, width, "lecun") * factor
            weight = Tensor(
                np.random.default_rng(layer_index).normal(0.0, scale, (width, width))
            )
            x = (x @ weight).tanh()
            stds.append(float(np.std(x.data)))
        return stds

    def test_a_correct_scale_keeps_activations_alive(self):
        final = self.depth_profile(1.0)[-1]
        self.assertGreater(final, 0.05)
        self.assertLess(final, 1.0)

    def test_too_small_a_scale_vanishes(self):
        self.assertLess(self.depth_profile(0.1)[-1], 1e-6)

    def test_too_large_a_scale_saturates(self):
        # tanh pinned near +/-1 everywhere: std approaches 1 and the
        # gradient (1 - tanh^2) approaches 0.
        self.assertGreater(self.depth_profile(6.0)[-1], 0.85)

    def test_vanishing_is_monotonic(self):
        profile = self.depth_profile(0.1, depth=10)
        self.assertEqual(profile, sorted(profile, reverse=True))


class TestSymmetryBreaking(unittest.TestCase):
    """Zero weights are the one initialization that cannot work."""

    def test_identical_weights_receive_identical_gradients(self):
        layer = Linear(4, 3, seed=0)
        layer.weight = Tensor(np.zeros((4, 3)))
        x = Tensor(np.random.default_rng(0).normal(size=(6, 4)))
        ((layer(x)) ** 2).sum().backward()
        columns = layer.weight.grad.T
        np.testing.assert_allclose(columns[0], columns[1])
        np.testing.assert_allclose(columns[1], columns[2])

    def test_random_weights_do_not(self):
        layer = Linear(4, 3, seed=0)
        x = Tensor(np.random.default_rng(0).normal(size=(6, 4)))
        ((layer(x)) ** 2).sum().backward()
        columns = layer.weight.grad.T
        self.assertFalse(np.allclose(columns[0], columns[1]))


class TestLinear(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(Linear(4, 3, seed=0)(Tensor(np.zeros((5, 4)))).shape, (5, 3))

    def test_extra_leading_dimensions_are_preserved(self):
        # (batch, time, features) is what attention will feed it.
        self.assertEqual(Linear(4, 3, seed=0)(Tensor(np.zeros((2, 6, 4)))).shape, (2, 6, 3))

    def test_bias_starts_at_zero(self):
        np.testing.assert_array_equal(Linear(5, 3, seed=0).bias.data, np.zeros(3))

    def test_weights_do_not_start_at_zero(self):
        self.assertGreater(float(np.abs(Linear(5, 3, seed=0).weight.data).sum()), 0.0)

    def test_weight_std_matches_the_requested_init(self):
        layer = Linear(256, 256, init="he", seed=0)
        self.assertAlmostEqual(float(layer.weight.data.std()),
                               init_scale(256, 256, "he"), delta=0.01)

    def test_seed_is_reproducible(self):
        np.testing.assert_array_equal(
            Linear(4, 3, seed=7).weight.data, Linear(4, 3, seed=7).weight.data
        )

    def test_different_seeds_differ(self):
        self.assertFalse(np.array_equal(
            Linear(4, 3, seed=1).weight.data, Linear(4, 3, seed=2).weight.data
        ))

    def test_zero_input_gives_the_bias(self):
        layer = Linear(4, 3, seed=0)
        layer.bias = Tensor([1.0, -2.0, 0.5])
        np.testing.assert_allclose(layer(Tensor(np.zeros((2, 4)))).data,
                                   np.tile([1.0, -2.0, 0.5], (2, 1)))

    def test_wrong_input_width_is_named(self):
        with self.assertRaises(ValueError) as caught:
            Linear(4, 3, seed=0)(Tensor(np.zeros((2, 5))))
        self.assertIn("expected last dimension 4", str(caught.exception))

    def test_non_positive_features_rejected(self):
        with self.assertRaises(ValueError):
            Linear(0, 3)


class TestSequential(unittest.TestCase):
    def test_applies_in_order(self):
        net = Sequential(Linear(3, 4, seed=0), Tanh(), Linear(4, 2, seed=1))
        self.assertEqual(net(Tensor(np.zeros((5, 3)))).shape, (5, 2))

    def test_activation_is_applied(self):
        net = Sequential(Linear(3, 4, seed=0), Tanh())
        out = net(Tensor(np.full((5, 3), 10.0)))
        self.assertLessEqual(float(np.abs(out.data).max()), 1.0)

    def test_relu_clips_negatives(self):
        out = ReLU()(Tensor([-2.0, 3.0]))
        np.testing.assert_allclose(out.data, [0.0, 3.0])

    def test_empty_sequential_is_the_identity(self):
        x = Tensor([[1.0, 2.0]])
        np.testing.assert_allclose(Sequential()(x).data, x.data)


class TestGradients(unittest.TestCase):
    """Still checked against finite differences, as every day is."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_linear(self):
        check_gradient(lambda x, w, b: (x @ w + b).sum(),
                       [self.rng.normal(size=(4, 3)), self.rng.normal(size=(3, 2)),
                        self.rng.normal(size=(2,))])

    def test_linear_with_activation(self):
        check_gradient(lambda x, w, b: (x @ w + b).tanh().sum(),
                       [self.rng.normal(size=(4, 3)), self.rng.normal(size=(3, 2)),
                        self.rng.normal(size=(2,))])

    def test_two_layers(self):
        check_gradient(lambda x, w1, w2: ((x @ w1).tanh() @ w2).sum(),
                       [self.rng.normal(size=(4, 3)), self.rng.normal(size=(3, 5)),
                        self.rng.normal(size=(5, 2))])

    def test_a_real_module_stack_produces_gradients_everywhere(self):
        net = Sequential(Linear(3, 5, seed=0), Tanh(), Linear(5, 2, seed=1))
        (net(Tensor(self.rng.normal(size=(6, 3)))) ** 2).sum().backward()
        for parameter in net.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)

    def test_one_step_reduces_the_loss(self):
        rng = np.random.default_rng(1)
        x = Tensor(rng.normal(size=(16, 3)))
        target = Tensor(rng.normal(size=(16, 2)))
        net = Sequential(Linear(3, 8, seed=0), Tanh(), Linear(8, 2, seed=1))
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        before = float(loss.data)
        for parameter in net.parameters():
            parameter.data = parameter.data - 0.1 * parameter.grad
        after = float(((net(x) - target) ** 2).mean().data)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
