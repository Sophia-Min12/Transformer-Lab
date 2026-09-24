# Day 10 · Causal Masking

> Day 9 ended by noting that position 0 was attending to positions 1–4. For a classifier that is correct. For a model trained to predict the next token it is **fatal** — the answer is sitting in the input, so the model learns to copy it and the loss goes to zero without a single thing being learned about language.

## The fix needs no new machinery

A matrix that is zero on and below the diagonal and hugely negative above it, added to the scores **before** the softmax:

```
[[0, -1e9, -1e9],
 [0,    0, -1e9],
 [0,    0,    0]]
```

The forbidden entries exponentiate to zero and the survivors renormalize themselves, so nothing downstream knows a mask happened:

```
position 0: 1.00 0.00 0.00 0.00 0.00
position 1: 0.50 0.50 0.00 0.00 0.00
position 2: 0.33 0.33 0.33 0.00 0.00
position 3: 0.25 0.25 0.25 0.25 0.00
position 4: 0.20 0.20 0.20 0.20 0.20
```

**Every row still sums to 1.** The mask removes options, not probability mass — a test pins that, because a mask applied *after* the softmax would not have this property and would quietly rescale everything.

The mask also adds **nothing to learn**: a test asserts `CausalSelfAttention` and `SelfAttention` have identical parameter counts.

## Test the gradient, not the weights

This is the part worth taking from the day. A zero attention weight is *suggestive*. A zero **gradient** is proof that changing an input cannot change an output:

```
output position -> input positions it depends on
  0 -> [0]
  1 -> [0, 1]
  2 -> [0, 1, 2]
  3 -> [0, 1, 2, 3]
  4 -> [0, 1, 2, 3, 4]
strictly lower-triangular: True
```

Without the mask, the same probe gives:

```
  0 -> [0, 1, 2, 3, 4]
```

Position 0 depends on the future.

`depends_on()` runs a backward pass per output row and reports which inputs received gradient. It **cannot be fooled** by a mask applied to the wrong axis — and a test proves that by transposing the mask, which still prints as a triangle while letting the future straight through. Shape checks and eyeballed weight matrices both miss that; the gradient does not.

## Why `-1e9` and not `-inf`

```
with -1e9, a causal mask gives   [[1.0, 0.0], [0.268941, 0.731059]]
a fully masked row with -inf     [[nan, nan, nan]]
the same row with -1e9           [[0.333333, 0.333333, 0.333333]]
```

`-inf` works perfectly — until a row is *entirely* masked, which happens routinely with padding masks where a padded query attends to nothing. Then every score in the row is `-inf`, Day 6's stabilizing shift subtracts `-inf`, and **`-inf − -inf` is `nan`**. One `nan` poisons the whole batch.

A large finite value degrades to a uniform row instead. The output is meaningless, but it is *finite* meaningless, and the rest of the batch survives. Both behaviours are pinned by tests.

## Run it

```bash
# demo (the mask, weights, the gradient probe, the -inf failure)
python masking.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_attention/day10_causal_masking

# tests — from inside this folder
python -m unittest
python test_masking.py

# the docstring examples are runnable too
python -m doctest masking.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

One attention layer computes **one** set of weights per position, so it can express one relationship at a time — "the adjective before me" or "the subject of the sentence", not both. **Day 11** runs several in parallel on slices of the same vectors, which costs nothing extra because Day 4's matmul already batches.
