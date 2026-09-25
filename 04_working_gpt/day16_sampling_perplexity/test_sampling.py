"""Day 16 tests - decoders, the filters, and what perplexity does not say.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from sampling import (
    CORPUS,
    NOVEL_SENTENCES,
    CharNgram,
    CharTokenizer,
    GPT,
    distinct_ratio,
    generate,
    perplexity,
    probabilities,
    sample_next,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    train_gpt,
)

LOGITS = np.array([2.0, 1.0, 0.5, -1.0, -3.0])


class TestTemperature(unittest.TestCase):
    def test_it_does_not_reorder_the_candidates(self):
        # The property that makes temperature safe: it changes the gaps,
        # never the ranking.
        ranking = np.argsort(probabilities(LOGITS, 1.0))
        for temperature in (0.1, 0.5, 2.0, 10.0):
            np.testing.assert_array_equal(np.argsort(probabilities(LOGITS, temperature)),
                                          ranking)

    def test_a_low_temperature_concentrates_the_mass(self):
        self.assertGreater(probabilities(LOGITS, 0.25).max(), probabilities(LOGITS, 1.0).max())

    def test_a_high_temperature_spreads_it(self):
        self.assertLess(probabilities(LOGITS, 5.0).max(), probabilities(LOGITS, 1.0).max())

    def test_entropy_increases_with_temperature(self):
        entropies = []
        for temperature in (0.25, 0.5, 1.0, 2.0, 5.0):
            probs = probabilities(LOGITS, temperature)
            entropies.append(-float((probs * np.log(probs)).sum()))
        self.assertEqual(entropies, sorted(entropies))

    def test_a_very_high_temperature_approaches_uniform(self):
        probs = probabilities(LOGITS, 1e6)
        np.testing.assert_allclose(probs, np.full(5, 0.2), atol=1e-5)

    def test_probabilities_sum_to_one(self):
        for temperature in (0.1, 1.0, 7.0):
            self.assertAlmostEqual(float(probabilities(LOGITS, temperature).sum()), 1.0)

    def test_it_is_stable_on_large_logits(self):
        # exp(800) overflows; the max is subtracted first.
        probs = probabilities(np.array([800.0, 799.0, 0.0]), 1.0)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_negative_temperature_rejected(self):
        with self.assertRaises(ValueError):
            temperature_scale(LOGITS, -1.0)

    def test_zero_is_rejected_here_and_handled_in_sample_next(self):
        with self.assertRaises(ValueError):
            temperature_scale(LOGITS, 0.0)
        self.assertEqual(sample_next(LOGITS, temperature=0), 0)


class TestGreedy(unittest.TestCase):
    def test_it_is_the_argmax(self):
        self.assertEqual(sample_next(LOGITS, temperature=0), int(np.argmax(LOGITS)))

    def test_it_is_deterministic(self):
        picks = {sample_next(LOGITS, temperature=0) for _ in range(20)}
        self.assertEqual(len(picks), 1)

    def test_a_very_low_temperature_converges_to_it(self):
        rng = np.random.default_rng(0)
        picks = {sample_next(LOGITS, temperature=0.01, rng=rng) for _ in range(50)}
        self.assertEqual(picks, {int(np.argmax(LOGITS))})


class TestTopK(unittest.TestCase):
    def test_it_keeps_exactly_k_candidates(self):
        self.assertEqual(int((top_k_filter(probabilities(LOGITS), 2) > 0).sum()), 2)

    def test_the_kept_ones_are_the_most_likely(self):
        kept = top_k_filter(probabilities(LOGITS), 2)
        np.testing.assert_array_equal(np.nonzero(kept)[0], [0, 1])

    def test_it_renormalizes(self):
        self.assertAlmostEqual(float(top_k_filter(probabilities(LOGITS), 3).sum()), 1.0)

    def test_k_of_one_is_greedy(self):
        kept = top_k_filter(probabilities(LOGITS), 1)
        self.assertEqual(int(np.argmax(kept)), 0)
        self.assertAlmostEqual(float(kept.max()), 1.0)

    def test_k_beyond_the_vocabulary_is_a_no_op(self):
        probs = probabilities(LOGITS)
        np.testing.assert_allclose(top_k_filter(probs, 99), probs)

    def test_the_tail_can_never_be_sampled(self):
        rng = np.random.default_rng(0)
        picks = {sample_next(LOGITS, 1.0, top_k=2, rng=rng) for _ in range(200)}
        self.assertTrue(picks <= {0, 1})

    def test_k_below_one_rejected(self):
        with self.assertRaises(ValueError):
            top_k_filter(probabilities(LOGITS), 0)


class TestTopP(unittest.TestCase):
    def test_it_keeps_enough_mass(self):
        probs = probabilities(LOGITS)
        kept = np.nonzero(top_p_filter(probs, 0.9))[0]
        self.assertGreaterEqual(float(probs[kept].sum()), 0.9)

    def test_it_keeps_no_more_than_needed(self):
        # Dropping the least likely kept token would fall below p.
        probs = probabilities(LOGITS)
        kept = np.nonzero(top_p_filter(probs, 0.9))[0]
        trimmed = sorted(probs[kept], reverse=True)[:-1]
        self.assertLess(sum(trimmed), 0.9)

    def test_it_adapts_where_top_k_cannot(self):
        # The argument for nucleus sampling, as a measurement: one fixed
        # k cannot be right for both a confident and an unsure position.
        confident = probabilities(np.array([10.0, 1.0, 0.9, 0.8, 0.7]))
        unsure = probabilities(np.array([1.0, 0.95, 0.9, 0.85, 0.8]))
        kept_confident = int((top_p_filter(confident, 0.9) > 0).sum())
        kept_unsure = int((top_p_filter(unsure, 0.9) > 0).sum())
        self.assertLess(kept_confident, kept_unsure)

    def test_it_renormalizes(self):
        self.assertAlmostEqual(float(top_p_filter(probabilities(LOGITS), 0.5).sum()), 1.0)

    def test_p_of_one_keeps_everything(self):
        probs = probabilities(LOGITS)
        np.testing.assert_allclose(top_p_filter(probs, 1.0), probs)

    def test_a_dominant_token_degenerates_to_greedy_not_to_empty(self):
        probs = probabilities(np.array([20.0, 0.0, 0.0]))
        kept = top_p_filter(probs, 0.5)
        self.assertEqual(int((kept > 0).sum()), 1)
        self.assertAlmostEqual(float(kept[0]), 1.0)

    def test_bad_p_rejected(self):
        for p in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                top_p_filter(probabilities(LOGITS), p)


class TestSampling(unittest.TestCase):
    def test_the_seed_makes_it_reproducible(self):
        a = [sample_next(LOGITS, 1.0, rng=np.random.default_rng(3)) for _ in range(10)]
        b = [sample_next(LOGITS, 1.0, rng=np.random.default_rng(3)) for _ in range(10)]
        self.assertEqual(a, b)

    def test_it_is_not_deterministic_without_a_low_temperature(self):
        rng = np.random.default_rng(0)
        self.assertGreater(len({sample_next(LOGITS, 1.5, rng=rng) for _ in range(100)}), 1)

    def test_empirical_frequencies_match_the_distribution(self):
        rng = np.random.default_rng(0)
        draws = [sample_next(LOGITS, 1.0, rng=rng) for _ in range(20000)]
        observed = np.bincount(draws, minlength=5) / len(draws)
        np.testing.assert_allclose(observed, probabilities(LOGITS), atol=0.02)

    def test_the_filters_compose(self):
        rng = np.random.default_rng(0)
        picks = {sample_next(LOGITS, 1.0, top_k=3, top_p=0.5, rng=rng) for _ in range(200)}
        self.assertTrue(picks <= {0, 1, 2})


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharTokenizer(CORPUS + NOVEL_SENTENCES)
        cls.model = GPT(vocab_size=cls.tokenizer.vocab_size, dim=16, blocks=1,
                        heads=2, seed=0)

    def test_it_returns_the_prompt_plus_length_characters(self):
        text = generate(self.model, self.tokenizer, "the ", length=20, seed=0)
        self.assertEqual(len(text), 24)
        self.assertTrue(text.startswith("the "))

    def test_greedy_is_reproducible_without_a_seed(self):
        a = generate(self.model, self.tokenizer, "the ", 20, temperature=0)
        b = generate(self.model, self.tokenizer, "the ", 20, temperature=0)
        self.assertEqual(a, b)

    def test_the_seed_reproduces_a_sample(self):
        a = generate(self.model, self.tokenizer, "the ", 20, temperature=1.0, seed=5)
        b = generate(self.model, self.tokenizer, "the ", 20, temperature=1.0, seed=5)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = generate(self.model, self.tokenizer, "the ", 40, temperature=1.0, seed=1)
        b = generate(self.model, self.tokenizer, "the ", 40, temperature=1.0, seed=2)
        self.assertNotEqual(a, b)

    def test_it_only_emits_characters_in_the_vocabulary(self):
        text = generate(self.model, self.tokenizer, "the ", 60, temperature=1.5, seed=0)
        self.assertTrue(set(text) <= set(self.tokenizer.characters))

    def test_a_long_generation_stays_inside_the_context(self):
        # More characters than max_length; the context is a sliding window.
        model = GPT(vocab_size=self.tokenizer.vocab_size, dim=16, blocks=1,
                    heads=2, max_length=16, seed=0)
        self.assertEqual(len(generate(model, self.tokenizer, "the ", 40, seed=0)), 44)


class TestDistinctRatio(unittest.TestCase):
    def test_a_loop_scores_near_zero(self):
        self.assertLess(distinct_ratio("abcd" * 50, 4), 0.05)

    def test_varied_text_scores_near_one(self):
        rng = np.random.default_rng(0)
        text = "".join(rng.choice(list("abcdefghij"), size=400))
        self.assertGreater(distinct_ratio(text, 4), 0.9)

    def test_text_shorter_than_n_is_one(self):
        self.assertEqual(distinct_ratio("ab", 4), 1.0)

    def test_temperature_raises_diversity(self):
        tokenizer = CharTokenizer(CORPUS)
        model = GPT(vocab_size=tokenizer.vocab_size, dim=16, blocks=1, heads=2, seed=0)
        train_gpt(model, tokenizer.encode(CORPUS), steps=40, length=16, batch=4, lr=0.02)
        cold = distinct_ratio(generate(model, tokenizer, "the ", 200, temperature=0), 4)
        hot = distinct_ratio(generate(model, tokenizer, "the ", 200,
                                      temperature=1.5, seed=7), 4)
        self.assertGreater(hot, cold)

    def test_the_corpus_is_more_repetitive_than_any_sample(self):
        # The caveat the demo states: this metric measures diversity, not
        # quality. CORPUS is twelve copies of five sentences, so "closer
        # to the corpus" would mean *more* repetitive.
        self.assertLess(distinct_ratio(CORPUS, 4), 0.1)


class TestPerplexity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharTokenizer(CORPUS + NOVEL_SENTENCES)
        cls.ids = cls.tokenizer.encode(CORPUS)
        # Well enough trained for the measurements below to mean
        # something: a model still near uniform shows no context effect
        # and no penalty past its window, which is how the first version
        # of these tests failed.
        cls.model = GPT(vocab_size=cls.tokenizer.vocab_size, dim=24, blocks=1,
                        heads=2, seed=0)
        cls.history = train_gpt(cls.model, cls.ids, steps=150, length=16,
                                batch=4, lr=0.02, seed=0)

    def test_an_untrained_model_scores_about_the_vocabulary_size(self):
        fresh = GPT(vocab_size=self.tokenizer.vocab_size, dim=16, blocks=1,
                    heads=2, seed=0)
        self.assertAlmostEqual(perplexity(fresh, self.ids, window=16),
                               self.tokenizer.vocab_size, delta=3.0)

    def test_training_lowers_it_well_below_uniform(self):
        self.assertLess(perplexity(self.model, self.ids, window=16),
                        0.5 * self.tokenizer.vocab_size)

    def test_it_is_the_exponential_of_the_mean_nll(self):
        ids = self.ids[:17]
        logits = self.model(ids[:-1]).data
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        expected = math.exp(-log_probs[np.arange(16), ids[1:]].mean())
        self.assertAlmostEqual(perplexity(self.model, ids, window=16), expected, places=9)

    def test_a_single_token_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            perplexity(self.model, self.ids[:1], window=16)

    def test_more_context_helps_up_to_the_trained_window(self):
        scores = [perplexity(self.model, self.ids, window=w) for w in (4, 8, 16)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestPositionsNeverTrained(unittest.TestCase):
    """The day's sharpest finding, and it arrived as a bug.

    A sinusoidal encoding is defined at every position, so evaluating
    past the training window raises nothing - it just returns a much
    worse number, which looks like a bad model rather than a bad
    measurement.
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharTokenizer(CORPUS)
        cls.ids = cls.tokenizer.encode(CORPUS)
        cls.short = GPT(vocab_size=cls.tokenizer.vocab_size, dim=24, blocks=1,
                        heads=2, seed=0)
        train_gpt(cls.short, cls.ids, steps=150, length=16, batch=4, lr=0.02, seed=0)

    def test_loss_is_far_worse_just_past_the_training_window(self):
        # Compared at the boundary, sixteen positions either side. Further
        # out the profile is not monotone at this model size - positions
        # 16-23 score 3.00 and 40-47 recover to 1.77 - so asserting a
        # steady decline would be asserting noise. The demo's larger
        # model does decline steadily; this one only shows the step.
        ids = self.ids[:49]
        logits = self.short(ids[:-1]).data
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        nll = -log_probs[np.arange(48), ids[1:]]
        self.assertGreater(nll[16:32].mean(), 1.5 * nll[:16].mean())

    def test_perplexity_degrades_beyond_it(self):
        inside = perplexity(self.short, self.ids, window=16)
        outside = perplexity(self.short, self.ids, window=64)
        self.assertGreater(outside, 2.0 * inside)

    def test_training_on_a_longer_window_fixes_it(self):
        longer = GPT(vocab_size=self.tokenizer.vocab_size, dim=24, blocks=1,
                     heads=2, seed=0)
        train_gpt(longer, self.ids, steps=150, length=48, batch=4, lr=0.02, seed=0)
        self.assertLess(perplexity(longer, self.ids, window=48),
                        perplexity(self.short, self.ids, window=48))


class TestNgramBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharTokenizer(CORPUS + NOVEL_SENTENCES)
        cls.ids = cls.tokenizer.encode(CORPUS)
        cls.model = CharNgram(order=3, k=0.1).fit(cls.ids, cls.tokenizer.vocab_size)

    def test_it_beats_uniform_by_a_wide_margin(self):
        self.assertLess(self.model.perplexity(self.ids), 0.2 * self.tokenizer.vocab_size)

    def test_it_generalizes_to_unseen_recombinations(self):
        novel = self.tokenizer.encode(NOVEL_SENTENCES)
        self.assertLess(self.model.perplexity(novel), 0.3 * self.tokenizer.vocab_size)

    def test_a_higher_order_fits_the_training_text_better(self):
        scores = [CharNgram(order=n, k=0.1).fit(self.ids, self.tokenizer.vocab_size)
                  .perplexity(self.ids) for n in (1, 2, 3, 4)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_distributions_are_normalized(self):
        for history in ([], [1], [1, 2], [1, 2, 3]):
            self.assertAlmostEqual(float(self.model.distribution(history).sum()), 1.0)

    def test_smoothing_keeps_an_unseen_continuation_possible(self):
        # Without add-k an unseen pair gives probability zero, and one
        # zero makes the whole perplexity infinite.
        self.assertTrue(np.all(self.model.distribution([0, 1]) > 0))

    def test_it_outscores_the_transformer_on_this_corpus(self):
        # The honest result of the day. A trigram needs no gradient, no
        # autograd and no seconds; on 2,940 characters of five repeated
        # sentences it wins, because there is no long-range structure
        # here for attention to buy.
        transformer = GPT(vocab_size=self.tokenizer.vocab_size, dim=24, blocks=1,
                          heads=2, seed=0)
        train_gpt(transformer, self.ids, steps=150, length=16, batch=4, lr=0.02, seed=0)
        self.assertLess(self.model.perplexity(self.ids),
                        perplexity(transformer, self.ids, window=16))

    def test_bad_arguments_rejected(self):
        with self.assertRaises(ValueError):
            CharNgram(order=0)
        with self.assertRaises(ValueError):
            CharNgram(order=3, k=0.0)


class TestTheSplitThatIsNotASplit(unittest.TestCase):
    """A random split of a repeated corpus measures nothing."""

    def test_the_tail_of_the_corpus_appears_in_its_head(self):
        # Which is why a held-out score on it looks like generalization
        # and is not: the model trained on this exact text.
        split = int(len(CORPUS) * 0.8)
        tail = CORPUS[split:split + 60]
        self.assertIn(tail, CORPUS[:split])

    def test_the_novel_sentences_do_not(self):
        for sentence in NOVEL_SENTENCES.split(". "):
            if len(sentence) > 12:
                self.assertNotIn(sentence.strip(), CORPUS)

    def test_they_use_only_the_training_vocabulary(self):
        # A fair test of recombination needs no unseen characters.
        self.assertTrue(set(NOVEL_SENTENCES) <= set(CORPUS))


if __name__ == "__main__":
    unittest.main()
