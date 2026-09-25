"""Day 15 tests - the tokenizer, the assembled model, and that it learns.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from gpt import (
    CORPUS,
    CharTokenizer,
    GPT,
    Tensor,
    check_gradient,
    cross_entropy,
    sequence_batches,
    train_gpt,
)


class TestCharTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = CharTokenizer(CORPUS)
        self.assertEqual(tokenizer.decode(tokenizer.encode("the cat sat")), "the cat sat")

    def test_vocabulary_is_the_distinct_characters(self):
        self.assertEqual(CharTokenizer("abcabc").vocab_size, 3)

    def test_the_vocabulary_is_sorted_so_ids_are_reproducible(self):
        # Rebuilding the tokenizer must give the same ids, or a saved
        # model decodes to noise.
        self.assertEqual(CharTokenizer("zyx").characters, ["x", "y", "z"])
        self.assertEqual(list(CharTokenizer("cba").encode("abc")), [0, 1, 2])

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            CharTokenizer("")

    def test_an_unknown_character_is_named_not_silently_dropped(self):
        with self.assertRaises(KeyError):
            CharTokenizer("abc").encode("abz")

    def test_encode_returns_integers(self):
        self.assertTrue(np.issubdtype(CharTokenizer("abc").encode("cab").dtype, np.integer))


class TestModelShape(unittest.TestCase):
    def test_logits_are_one_row_per_position(self):
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
        self.assertEqual(model(np.array([1, 2, 3])).shape, (3, 10))

    def test_a_two_dimensional_input_is_rejected(self):
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
        with self.assertRaises(ValueError):
            model(np.zeros((2, 3), dtype=int))

    def test_a_sequence_beyond_max_length_is_rejected(self):
        # The positional table has a fixed height; running past it would
        # otherwise fail somewhere deeper and less legibly.
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, max_length=4, seed=0)
        with self.assertRaises(ValueError):
            model(np.zeros(5, dtype=int))

    def test_bad_construction_arguments_rejected(self):
        for kwargs in ({"vocab_size": 0}, {"dim": 0}, {"blocks": 0}):
            with self.assertRaises(ValueError):
                GPT(**{"vocab_size": 8, **kwargs})


class TestUntrainedLoss(unittest.TestCase):
    """Day 6's cheapest bug-catch, applied to the whole model."""

    def test_it_starts_at_chance(self):
        tokenizer = CharTokenizer(CORPUS)
        model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=2, heads=2, seed=0)
        loss = float(model.loss(tokenizer.encode(CORPUS[:64])).data)
        self.assertAlmostEqual(loss, math.log(tokenizer.vocab_size), delta=0.15)

    def test_a_larger_vocabulary_starts_higher(self):
        losses = []
        for vocab in (8, 64, 256):
            model = GPT(vocab_size=vocab, dim=16, blocks=1, heads=2, seed=0)
            ids = np.arange(20) % vocab
            losses.append(float(model.loss(ids).data))
        self.assertEqual(losses, sorted(losses))


class TestTheOffset(unittest.TestCase):
    """The one-line mistake that produces a low loss instead of an error."""

    def test_loss_pairs_position_t_with_token_t_plus_one(self):
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
        ids = np.array([1, 2, 3, 4, 5])
        expected = cross_entropy(model(ids[:-1]), ids[1:])
        self.assertAlmostEqual(float(model.loss(ids).data), float(expected.data), places=12)

    def test_the_off_by_one_version_is_a_different_number(self):
        # Guards against someone "simplifying" the slice back.
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
        ids = np.array([1, 2, 3, 4, 5])
        wrong = cross_entropy(model(ids[:-1]), ids[:-1])
        self.assertNotAlmostEqual(float(model.loss(ids).data), float(wrong.data), places=3)

    def test_predicting_the_input_is_the_easier_problem(self):
        # Why the bug hides: copying its own input is learnable, so the
        # loss falls and nothing looks wrong.
        tokenizer = CharTokenizer(CORPUS)
        ids = tokenizer.encode(CORPUS[:400])

        def final_loss(shift):
            model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=1,
                        heads=2, seed=0)
            history = []
            for _ in range(60):
                logits = model(ids[:32])
                targets = ids[1:33] if shift else ids[:32]
                loss = cross_entropy(logits, targets)
                for parameter in model.parameters():
                    parameter.grad = np.zeros_like(parameter.data)
                loss.backward()
                for parameter in model.parameters():
                    parameter.data -= 0.05 * parameter.grad
                history.append(float(loss.data))
            return history[-1]

        self.assertLess(final_loss(shift=False), final_loss(shift=True))

    def test_a_single_token_cannot_form_a_prediction(self):
        model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
        with self.assertRaises(ValueError):
            model.loss(np.array([3]))


class TestWeightTying(unittest.TestCase):
    def test_tying_saves_vocab_times_dim_parameters(self):
        tied = GPT(vocab_size=20, dim=32, blocks=2, heads=4, seed=0, tie_weights=True)
        untied = GPT(vocab_size=20, dim=32, blocks=2, heads=4, seed=0, tie_weights=False)
        self.assertEqual(untied.parameter_count() - tied.parameter_count(), 20 * 32)

    def test_the_shared_tensor_is_counted_once(self):
        # Day 5's parameter walk deduplicates by identity. If it did not,
        # the optimizer would apply two updates to one tensor per step.
        model = GPT(vocab_size=20, dim=8, blocks=1, heads=2, seed=0, tie_weights=True)
        shared = [p for p in model.parameters() if p is model.tokens.weight]
        self.assertEqual(len(shared), 1)

    def test_both_forms_produce_the_same_shape(self):
        for tie in (True, False):
            model = GPT(vocab_size=12, dim=8, blocks=1, heads=2, seed=0, tie_weights=tie)
            self.assertEqual(model(np.array([1, 2, 3, 4])).shape, (4, 12))

    def test_the_untied_head_gets_its_own_gradient(self):
        model = GPT(vocab_size=12, dim=8, blocks=1, heads=2, seed=0, tie_weights=False)
        model.loss(np.array([1, 2, 3, 4])).backward()
        self.assertGreater(float(np.abs(model.head.weight.grad).sum()), 0.0)


class TestTheFinalLayerNorm(unittest.TestCase):
    """Easy to leave out. What makes it matter is training, not depth.

    My first version of these tests asserted a 5x logit-scale ratio on a
    fresh model and failed at 2.86x, which sent me to measure where the
    ratio actually comes from:

        untrained, 32 wide:  1 block 1.7x   2 blocks 2.9x   16 blocks 4.2x
        the same 2 blocks, after 120 training steps:        22.9x

    So depth at initialization contributes mildly, and **training** is
    what grows the residual stream - eight times more than depth did, at
    a fixed depth. Day 13 measured growth with depth in a stack of random
    weights; it is learned weights that make this layer necessary.
    """

    @staticmethod
    def ratio(model):
        keep = model.final_norm
        with_norm = float(np.abs(model(np.arange(12) % 20).data).max())
        model.final_norm = lambda x: x
        without = float(np.abs(model(np.arange(12) % 20).data).max())
        model.final_norm = keep
        return without / with_norm

    def test_a_fresh_model_shows_only_a_small_ratio(self):
        self.assertLess(self.ratio(GPT(vocab_size=20, dim=32, blocks=2, heads=4, seed=0)), 6.0)

    def test_depth_widens_it_a_little(self):
        shallow = self.ratio(GPT(vocab_size=20, dim=32, blocks=1, heads=4, seed=0))
        deep = self.ratio(GPT(vocab_size=20, dim=32, blocks=16, heads=4, seed=0))
        self.assertGreater(deep, shallow)

    def test_training_widens_it_far_more(self):
        tokenizer = CharTokenizer(CORPUS)
        model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=1, heads=2, seed=0)
        before = self.ratio(model)
        train_gpt(model, tokenizer.encode(CORPUS), steps=40, length=16, batch=4, lr=0.02)
        after = self.ratio(model)
        self.assertGreater(after / before, 5.0)

    def test_with_the_norm_the_logits_stay_in_a_trainable_range(self):
        model = GPT(vocab_size=20, dim=32, blocks=2, heads=4, seed=0)
        self.assertLess(float(np.abs(model(np.arange(12) % 20).data).max()), 50.0)


class TestCausality(unittest.TestCase):
    """Perturbation, not gradient - see the note in the day's README."""

    def test_changing_a_token_leaves_every_earlier_logit_untouched(self):
        model = GPT(vocab_size=10, dim=16, blocks=2, heads=2, seed=0)
        ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        baseline = model(ids).data
        changed = ids.copy()
        changed[5] = 9
        perturbed = model(changed).data
        np.testing.assert_array_equal(baseline[:5], perturbed[:5])

    def test_and_does_change_the_logits_from_that_position_on(self):
        model = GPT(vocab_size=10, dim=16, blocks=2, heads=2, seed=0)
        ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        changed = ids.copy()
        changed[5] = 9
        self.assertFalse(np.allclose(model(ids).data[5:], model(changed).data[5:]))

    def test_the_gradient_probe_of_day_10_cannot_see_this_on_a_tied_model(self):
        # Pins the mistake the first version of this check made. With
        # tied weights the embedding table is the output projection too,
        # so every row receives gradient from the vocabulary softmax at
        # every position - regardless of causality. The probe reports a
        # near-full table either way, which is why perturbation replaced
        # it: position 0 can only have seen token ids[0], yet far more
        # than one row comes back non-zero.
        model = GPT(vocab_size=10, dim=16, blocks=1, heads=2, seed=0)
        ids = np.array([0, 1, 2, 3])
        logits = model(ids)
        selector = np.zeros((4, 10))
        selector[0] = 1.0
        (logits * Tensor(selector)).sum().backward()
        rows_touched = int((np.abs(model.tokens.weight.grad).sum(axis=-1) > 1e-12).sum())
        self.assertGreater(rows_touched, 1)


class TestTraining(unittest.TestCase):
    def test_the_loss_falls_well_below_chance(self):
        tokenizer = CharTokenizer(CORPUS)
        ids = tokenizer.encode(CORPUS)
        model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=1, heads=2, seed=0)
        history = train_gpt(model, ids, steps=40, length=16, batch=4, lr=0.02)
        self.assertLess(history[-1], 0.85 * math.log(tokenizer.vocab_size))

    def test_the_trend_is_downward_not_merely_the_last_step(self):
        tokenizer = CharTokenizer(CORPUS)
        ids = tokenizer.encode(CORPUS)
        model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=1, heads=2, seed=0)
        history = train_gpt(model, ids, steps=40, length=16, batch=4, lr=0.02)
        self.assertLess(float(np.mean(history[-10:])), float(np.mean(history[:10])))

    def test_every_parameter_receives_gradient(self):
        model = GPT(vocab_size=20, dim=16, blocks=2, heads=2, seed=0)
        model.loss(np.arange(12) % 20).backward()
        for parameter in model.parameters():
            self.assertGreater(float(np.abs(parameter.grad).sum()), 0.0)
            self.assertTrue(np.all(np.isfinite(parameter.grad)))


class TestSequenceBatches(unittest.TestCase):
    def test_each_window_is_length_plus_one(self):
        windows = sequence_batches(np.arange(100), length=8, count=5)
        self.assertEqual([len(w) for w in windows], [9] * 5)

    def test_the_count_is_honoured(self):
        self.assertEqual(len(sequence_batches(np.arange(100), length=8, count=7)), 7)

    def test_a_corpus_shorter_than_a_window_is_rejected(self):
        with self.assertRaises(ValueError):
            sequence_batches(np.arange(5), length=8, count=2)

    def test_the_seed_makes_it_reproducible(self):
        a = sequence_batches(np.arange(100), length=8, count=4, seed=3)
        b = sequence_batches(np.arange(100), length=8, count=4, seed=3)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)


class TestGradients(unittest.TestCase):
    """The whole model against finite differences, not just its parts."""

    def test_the_loss_gradient_at_the_embedding_table(self):
        rng = np.random.default_rng(0)
        model = GPT(vocab_size=6, dim=8, blocks=1, heads=2, seed=0)
        ids = np.array([1, 3, 2, 5, 0])

        def forward(table):
            model.tokens.weight = table
            return model.loss(ids)

        check_gradient(forward, [rng.normal(size=(6, 8)) * 0.1])

    def test_the_loss_gradient_at_a_block_weight(self):
        rng = np.random.default_rng(1)
        model = GPT(vocab_size=6, dim=8, blocks=1, heads=2, seed=0, tie_weights=False)
        ids = np.array([1, 3, 2, 5, 0])

        def forward(weight):
            model.blocks[0].feedforward.up.weight = weight
            return model.loss(ids)

        check_gradient(forward, [rng.normal(size=(8, 32)) * 0.1])


if __name__ == "__main__":
    unittest.main()
