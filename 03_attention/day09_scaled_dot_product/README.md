# Day 9 · Scaled Dot-Product Attention

> Every layer so far applies weights that were learned once and then held fixed — a `Linear` treats the thousandth example exactly like the first. Attention is the first operation where **the coefficients are computed from the input itself**, freshly, for every position of every example.

```
attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

`Q` asks, `K` advertises, `V` carries the content, and the softmax turns the match between question and advertisement into mixing weights that sum to one. **The mechanism has no parameters at all** — everything learned lives in the three projections that produce Q, K and V. A test pins that a `SelfAttention(dim=8)` has exactly `3 × 8 × 8` parameters.

## Why `√d_k`, measured three ways

The dot product of two independent unit-variance vectors has variance `d_k`, so its **standard deviation grows as `√d_k`**:

```
   d_k   sd unscaled   sd scaled   sqrt(d_k)
     4          2.04        1.02        2.00
    16          3.93        0.98        4.00
    64          8.10        1.01        8.00
   256         16.03        1.00       16.00
  1024         32.31        1.01       32.00
```

The unscaled column tracks `√d_k` to two decimals. That is the whole reason the constant is `√d_k` and not something tuned.

**Consequence one — the attention distribution collapses.** Over 64 positions:

```
   d_k   entropy unscaled   entropy scaled   uniform
     4             2.8604           3.7033    4.1589
    64             0.7273           3.7114    4.1589
  1024             0.1816           3.6807    4.1589
```

Unscaled, entropy falls to **0.18 nats** — the row has collapsed onto a single position. That is a hard lookup, not a weighted average, and it happens before any training has taken place.

**Consequence two — the softmax stops responding.**

```
   d_k   sensitivity unscaled    scaled
     4                0.82999   0.93664
    64                0.27224   0.92486
  1024                0.07133   0.92977
```

This is `Σⱼ pⱼ(1−pⱼ)`, the diagonal of the softmax Jacobian: how much the weights move when a score moves. Unscaled it collapses **12×**; the scores can change and the output barely will, so the Q and K projections receive almost no signal. Scaled, it is flat at ~0.92 across two orders of magnitude of `d_k`.

### The measurement I got wrong first

My first attempt probed `|∂out/∂query|` and found it **larger** without scaling — the opposite of the claim. The probe was wrong, not the claim: the `1/√d_k` factor multiplies that gradient *directly*, so it conflates the scale with the saturation it is supposed to detect.

The softmax Jacobian has no such contamination. A test pins the backwards reading too, so the bad probe cannot quietly return as evidence.

## Properties worth testing

- **Weights are a distribution** — non-negative, rows sum to 1.
- **The output is a convex combination of the values**, so it must lie inside their range. This catches sign and transpose errors that shape checks miss entirely.
- **Identical keys give uniform attention** — a closed form to check against.
- **A dominant score selects one value exactly.**
- **A mask of zeros changes nothing**, which is what makes Day 10's mask a pure addition.

## Every position sees every other one

```
weight row 0: [0.2, 0.31, 0.283, 0.078, 0.129]
position 0 attends to positions [0, 1, 2, 3, 4]
```

Which is correct here, and **fatal for a language model**: position 0 is reading the future. **Day 10** is about removing exactly that, and it turns out to need no new machinery — only a matrix of `−∞` added before the softmax.

## Run it

```bash
# demo (the scale, three measurements, self-attention)
python attention.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_attention/day09_scaled_dot_product

# tests — from inside this folder
python -m unittest
python test_attention.py

# the docstring examples are runnable too
python -m doctest attention.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

**Day 10** forbids looking ahead. **Day 11** runs several attentions in parallel and shows why one is not enough. **Day 12** points out that nothing so far knows what *order* the positions are in — attention is permutation-equivariant, and a language model that cannot tell "dog bites man" from "man bites dog" is not a language model.
