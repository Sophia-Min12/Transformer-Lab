"""Day 7 tests — the update rules, bias correction, and a loop that learns.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from optimizers import (
    Adam,
    Linear,
    SGD,
    Sequential,
    Tanh,
    Tensor,
    accuracy,
    cross_entropy,
    train,
    xor_dataset,
)


def _param(value, gradient):
    p = Tensor(value)
    p.grad = np.asarray(gradient, dtype=float)
    return p


class TestOptimizerBase(unittest.TestCase):
    def test_empty_parameter_list_rejected(self):
        # Almost always means Module.parameters() missed something.
        with self.assertRaises(ValueError):
            SGD([], lr=0.1)

    def test_non_positive_lr_rejected(self):
        with self.assertRaises(ValueError):
            SGD([Tensor([1.0])], lr=0.0)

    def test_zero_grad_clears(self):
        p = _param([1.0], [5.0])
        optimizer = SGD([p], lr=0.1)
        optimizer.zero_grad()
        self.assertEqual(float(p.grad[0]), 0.0)

    def test_step_count_increments(self):
        p = _param([1.0], [1.0])
        optimizer = SGD([p], lr=0.1)
        optimizer.step()
        optimizer.step()
        self.assertEqual(optimizer.steps, 2)


class TestSGD(unittest.TestCase):
    def test_plain_update(self):
        p = _param([1.0], [2.0])
        SGD([p], lr=0.1).step()
        self.assertAlmostEqual(float(p.data[0]), 0.8)

    def test_moves_against_the_gradient(self):
        for gradient, expected in ((1.0, -1), (-1.0, 1)):
            p = _param([0.0], [gradient])
            SGD([p], lr=0.1).step()
            self.assertEqual(np.sign(float(p.data[0])), expected)

    def test_zero_gradient_does_nothing(self):
        p = _param([3.0], [0.0])
        SGD([p], lr=0.5).step()
        self.assertAlmostEqual(float(p.data[0]), 3.0)

    def test_momentum_accumulates(self):
        # With a constant gradient the velocity grows toward g/(1-mu).
        p = _param([0.0], [1.0])
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        moves = []
        previous = 0.0
        for _ in range(5):
            p.grad = np.array([1.0])
            optimizer.step()
            moves.append(abs(float(p.data[0]) - previous))
            previous = float(p.data[0])
        self.assertEqual(moves, sorted(moves))

    def test_momentum_zero_matches_plain_sgd(self):
        a, b = _param([1.0], [2.0]), _param([1.0], [2.0])
        SGD([a], lr=0.1).step()
        SGD([b], lr=0.1, momentum=0.0).step()
        self.assertAlmostEqual(float(a.data[0]), float(b.data[0]))

    def test_momentum_out_of_range_rejected(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                SGD([Tensor([1.0])], lr=0.1, momentum=bad)


class TestAdam(unittest.TestCase):
    def test_first_step_is_exactly_lr(self):
        # With bias correction, m_hat/sqrt(v_hat) = sign(g) at t=1.
        p = _param([0.0], [5.0])
        Adam([p], lr=0.1).step()
        self.assertAlmostEqual(abs(float(p.data[0])), 0.1, places=6)

    def test_first_step_is_almost_independent_of_gradient_magnitude(self):
        # The defining property: the step is normalized by the gradient's
        # own running scale, so a gradient a million times larger produces
        # the same step.
        #
        # "Almost", because eps sits in the denominator and is not scaled
        # with the gradient - so it perturbs a small gradient relatively
        # more. At g=0.001 the step is short by ~1e-6. That is eps doing
        # exactly its job, not a flaw in the invariance.
        sizes = []
        for gradient in (0.001, 1.0, 1000.0):
            p = _param([0.0], [gradient])
            Adam([p], lr=0.1).step()
            sizes.append(abs(float(p.data[0])))
        for size in sizes:
            self.assertAlmostEqual(size, 0.1, places=4)
        self.assertLess(sizes[0], sizes[1])  # eps costs the small gradient a little

    def test_eps_is_what_breaks_the_exact_invariance(self):
        # With a much smaller eps the same comparison tightens by orders
        # of magnitude, which identifies eps as the cause.
        sizes = []
        for gradient in (0.001, 1.0):
            p = _param([0.0], [gradient])
            Adam([p], lr=0.1, eps=1e-16).step()
            sizes.append(abs(float(p.data[0])))
        self.assertAlmostEqual(sizes[0], sizes[1], places=10)

    def test_moves_against_the_gradient(self):
        for gradient, expected in ((2.0, -1), (-2.0, 1)):
            p = _param([0.0], [gradient])
            Adam([p], lr=0.1).step()
            self.assertEqual(np.sign(float(p.data[0])), expected)

    def test_bias_correction_changes_the_first_step(self):
        corrected, uncorrected = [], []
        for flag, target in ((True, corrected), (False, uncorrected)):
            p = _param([0.0], [1.0])
            Adam([p], lr=1.0, bias_correction=flag).step()
            target.append(abs(float(p.data[0])))
        self.assertAlmostEqual(corrected[0], 1.0, places=6)
        self.assertAlmostEqual(uncorrected[0], (1 - 0.9) / math.sqrt(1 - 0.999), places=4)

    def test_uncorrected_is_too_large_not_too_small(self):
        # The phrase "biased toward zero" suggests the opposite.
        corrected = _param([0.0], [1.0])
        uncorrected = _param([0.0], [1.0])
        Adam([corrected], lr=1.0, bias_correction=True).step()
        Adam([uncorrected], lr=1.0, bias_correction=False).step()
        self.assertGreater(abs(float(uncorrected.data[0])), abs(float(corrected.data[0])))

    def test_correction_keeps_the_step_at_lr_over_many_steps(self):
        p = _param([0.0], [1.0])
        optimizer = Adam([p], lr=0.1)
        previous = 0.0
        for _ in range(50):
            p.grad = np.array([1.0])
            optimizer.step()
            self.assertAlmostEqual(abs(float(p.data[0]) - previous), 0.1, places=3)
            previous = float(p.data[0])

    def test_bad_betas_rejected(self):
        for beta1, beta2 in ((1.0, 0.999), (0.9, 1.0), (-0.1, 0.999)):
            with self.assertRaises(ValueError):
                Adam([Tensor([1.0])], beta1=beta1, beta2=beta2)

    def test_non_positive_eps_rejected(self):
        with self.assertRaises(ValueError):
            Adam([Tensor([1.0])], eps=0.0)


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.x, self.y = xor_dataset(200, seed=0)

    def test_loss_decreases(self):
        net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        history = train(net, self.x, self.y, Adam(net.parameters(), lr=0.05), 200)
        self.assertLess(history[-1], history[0] / 10)

    def test_xor_is_learned(self):
        net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        train(net, self.x, self.y, Adam(net.parameters(), lr=0.05), 300)
        self.assertGreater(accuracy(net(self.x), self.y), 0.95)

    def test_a_linear_model_cannot_learn_xor(self):
        # Not a bug in the loop: no straight line separates XOR.
        linear = Sequential(Linear(2, 2, seed=0))
        train(linear, self.x, self.y, Adam(linear.parameters(), lr=0.05), 300)
        self.assertLess(accuracy(linear(self.x), self.y), 0.8)

    def test_starting_loss_is_about_log_two(self):
        net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        history = train(net, self.x, self.y, SGD(net.parameters(), lr=0.01), 1)
        self.assertAlmostEqual(history[0], math.log(2), delta=0.15)

    def test_history_length(self):
        net = Sequential(Linear(2, 4, seed=0))
        self.assertEqual(len(train(net, self.x, self.y, SGD(net.parameters()), 17)), 17)

    def test_zero_steps(self):
        net = Sequential(Linear(2, 4, seed=0))
        self.assertEqual(train(net, self.x, self.y, SGD(net.parameters()), 0), [])

    def test_negative_steps_rejected(self):
        net = Sequential(Linear(2, 4, seed=0))
        with self.assertRaises(ValueError):
            train(net, self.x, self.y, SGD(net.parameters()), -1)

    def test_every_parameter_actually_moves(self):
        net = Sequential(Linear(2, 8, seed=0), Tanh(), Linear(8, 2, seed=1))
        before = [p.data.copy() for p in net.parameters()]
        train(net, self.x, self.y, Adam(net.parameters(), lr=0.05), 20)
        for old, parameter in zip(before, net.parameters()):
            self.assertGreater(float(np.abs(parameter.data - old).sum()), 0.0)


class TestZeroGradMatters(unittest.TestCase):
    """Skipping it does not crash; it changes the effective learning rate."""

    def setUp(self):
        self.x, self.y = xor_dataset(200, seed=0)

    def run_loop(self, clear: bool, steps: int = 60):
        net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        optimizer = SGD(net.parameters(), lr=0.05)
        for _ in range(steps):
            loss = cross_entropy(net(self.x), self.y)
            if clear:
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return net

    def test_the_two_runs_differ(self):
        cleared = self.run_loop(True)
        kept = self.run_loop(False)
        self.assertFalse(np.allclose(cleared.parameters()[0].data,
                                     kept.parameters()[0].data))

    def test_gradients_really_do_accumulate_without_it(self):
        net = Sequential(Linear(2, 4, seed=0))
        first = cross_entropy(net(self.x), self.y)
        first.backward()
        after_one = net.parameters()[0].grad.copy()
        second = cross_entropy(net(self.x), self.y)
        second.backward()
        np.testing.assert_allclose(net.parameters()[0].grad, 2 * after_one, rtol=1e-9)


class TestAdamOnBadlyScaledFeatures(unittest.TestCase):
    """Where the per-parameter step size earns its keep."""

    def setUp(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(300, 2))
        raw[:, 1] *= 100.0
        self.x = Tensor(raw)
        self.y = (raw[:, 0] + raw[:, 1] * 0.01 > 0).astype(int)

    def fit(self, make_optimizer, steps=400):
        net = Sequential(Linear(2, 8, seed=0), Tanh(), Linear(8, 2, seed=1))
        train(net, self.x, self.y, make_optimizer(net.parameters()), steps)
        return accuracy(net(self.x), self.y)

    def test_adam_handles_the_scale_mismatch(self):
        self.assertGreater(self.fit(lambda p: Adam(p, lr=0.05)), 0.9)

    def test_sgd_struggles_at_a_safe_learning_rate(self):
        self.assertLess(self.fit(lambda p: SGD(p, lr=1e-4)), 0.85)

    def test_adam_beats_sgd_here(self):
        self.assertGreater(self.fit(lambda p: Adam(p, lr=0.05)),
                           self.fit(lambda p: SGD(p, lr=1e-4, momentum=0.9)))


class TestLearningRateStability(unittest.TestCase):
    def setUp(self):
        self.x, self.y = xor_dataset(200, seed=0)

    def losses_at(self, lr, steps=150):
        net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        return train(net, self.x, self.y, SGD(net.parameters(), lr=lr), steps)

    def test_a_sane_rate_descends_monotonically(self):
        losses = self.losses_at(0.1)
        uphill = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        self.assertEqual(uphill, 0)

    def test_a_huge_rate_goes_uphill(self):
        losses = self.losses_at(100.0)
        uphill = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        self.assertGreater(uphill, 0)

    def test_a_huge_rate_overshoots_far_above_the_start(self):
        losses = self.losses_at(100.0)
        self.assertGreater(max(losses), 10 * losses[0])

    def test_but_nothing_becomes_inf_or_nan(self):
        # tanh bounds every activation, so the network cannot blow up -
        # it only lands somewhere arbitrary.
        for lr in (10.0, 100.0):
            self.assertTrue(all(math.isfinite(v) for v in self.losses_at(lr)))


class TestXorDataset(unittest.TestCase):
    def test_shape_and_labels(self):
        x, y = xor_dataset(50, seed=0)
        self.assertEqual(x.shape, (50, 2))
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_deterministic(self):
        a, _ = xor_dataset(30, seed=1)
        b, _ = xor_dataset(30, seed=1)
        np.testing.assert_array_equal(a.data, b.data)

    def test_labels_follow_the_xor_of_the_signs(self):
        x, y = xor_dataset(400, noise=0.05, seed=0)
        expected = ((x.data[:, 0] > 0) ^ (x.data[:, 1] > 0)).astype(int)
        self.assertGreater(float(np.mean(expected == y)), 0.95)


if __name__ == "__main__":
    unittest.main()
