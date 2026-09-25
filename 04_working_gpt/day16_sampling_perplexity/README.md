# Day 16 · Sampling, Temperature, and Perplexity

> The model from Day 15 predicts. Today it writes, and today it is scored — against a trigram that has no parameters at all, and loses to it.

## Temperature changes the gaps, never the ranking

```
    T        'l'    'd'    'm'    'c'    't'   entropy
  0.25   0.929  0.038  0.017  0.016  0.000    0.333
  0.50   0.664  0.135  0.089  0.088  0.012    1.084
  1.00   0.395  0.178  0.145  0.144  0.053    1.707
  2.00   0.231  0.155  0.140  0.139  0.085    2.224
  5.00   0.121  0.103  0.099  0.099  0.081    2.738
                                    uniform = 2.996
```

Dividing the logits by `T` before the softmax sharpens or flattens the distribution and **cannot reorder it** — a test asserts the argsort is identical at every temperature. That is what makes temperature a safe knob: it changes how often the model takes its second choice, never what its second choice is. Greedy decoding is the `T → 0` limit, not a separate algorithm.

```
  greedy    'the cat the log the log the log theatheatheatheatheoe theheoehe '
  T = 0.5   'the cat the dog the log the dog atheatheathea theoeoe lheoehehe '
  T = 1.0   'the cat tog nd to. n saoundog the loheananana theoe theoeoeoeoei'
  T = 1.5   'the cat son lo tog ean rtogs asa ohearaladohe theoeoe. roriehe c'
```

**Top-k** keeps the `k` best candidates; **top-p** keeps the smallest set whose mass reaches `p`. The argument for the second is that `k` is fixed while the distribution is not, and a test measures exactly that: at `p = 0.9` a confident position keeps fewer tokens than an unsure one, which no fixed `k` can do.

## A metric that does not mean what it looks like

```
distinct 4-grams / total, over 200 generated characters

  greedy         0.49
  T = 0.5        0.56
  T = 1.0        0.89
  T = 1.5        0.95
  T=1, top-k 3   0.81
  T=1, top-p .9  0.79

  the training corpus itself   0.03
```

Diversity rises with temperature, as expected. But the corpus scores **0.03** — it is five sentences repeated twelve times, far more repetitive than anything the model emits. So "closer to the corpus" would mean *more* repetitive, and this number cannot be read as a quality score in either direction. It measures diversity, and that is all.

## The split that is not a split

```
perplexity, all at window 32

  training text          2.89
  'held out' last 20%    2.84
  genuinely unseen       3.08
```

The middle line is worthless, and it is worthless in the most dangerous way: it looks completely reasonable. `CORPUS` is five sentences repeated twelve times, so its last 20% is a *copy* of text the model trained on. Slicing it off and calling it held-out measures memorization and reports it as generalization.

The third line is the corpus's own words recombined into sentences that never occur in it — `the dog sat on the mat`, which the training text contains only as `the dog sat on the log` and `the cat sat on the mat`. A test asserts the tail of the corpus does appear in its head, and the novel sentences do not.

## The trigram wins

```
                        training    unseen
  trigram, add-0.1          1.65      1.75
  transformer, 25,856       2.89      3.08
  uniform                  20.00     20.00
```

A count-based trigram, no gradients and no parameters, beats the transformer on both splits and is not close. On 2,940 characters of five repeated sentences, two characters of history nearly determine the third — and that is precisely the structure an n-gram captures for free.

This is the honest state of the model rather than a bug to fix. The transformer's advantage is long-range structure, and **this corpus has none to sell**. A test pins the result so it cannot quietly reverse.

## A position never trained on is a position never learned

This arrived as a bug. Perplexity read 2.89 at window 32 and 11.27 at window 64, which made no sense until the training window turned out to be 32.

```
positions  0-15: mean nll 1.016
positions 16-31: mean nll 0.706
positions 32-47: mean nll 2.898   <- never trained here
positions 48-63: mean nll 4.880   <- never trained here
positions 64-79: mean nll 4.856   <- never trained here
positions 80-94: mean nll 6.188   <- never trained here
```

Day 12's sinusoidal encoding is *defined* at every position, so nothing raises an error — the model simply never saw a position past 31 and never learned to use one. Same corpus, same steps, only the training window changed:

```
  window          8     16     32     48     64     96
  trained on 32  2.89   2.87   2.89   5.83  11.27  26.49
  trained on 64  2.63   2.40   2.29   2.26   2.26   5.32
```

So a perplexity is comparable to another only at the same window, and meaningful only at or below the window the model trained on. `perplexity()` says so in its docstring, because nothing in the code can enforce it.

## Run it

```bash
# demo (temperature, the decoders, the baseline, the position finding)
python sampling.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_working_gpt/day16_sampling_perplexity

# tests — from inside this folder
python -m unittest
python test_sampling.py

# the docstring examples are runnable too
python -m doctest sampling.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Everything in this repo has been checked against finite differences and against itself, never against an implementation someone else wrote. **Day 17** puts the whole thing beside PyTorch, profiles where the NumPy version spends its time, and writes down what the comparison says — including the parts that do not flatter it.
