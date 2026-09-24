# Day 13 · LayerNorm and Residual Connections

> Levels 1–3 built every piece of a transformer block except the two that let you stack forty of them. Neither is glamorous and both are load-bearing.

## LayerNorm normalizes per position, not per batch

```
input  mean per row: [5.941, 4.5, -3.971]      output mean: [0.0, 0.0, -0.0]
input  std  per row: [3.694, 9.192, 7.199]     output std:  [1.0, 1.0, 1.0]
```

The statistics come from the **feature axis of each position**, so a token's output does not depend on which other tokens happen to share its batch. Training and single-sequence inference compute exactly the same thing — which BatchNorm cannot say, and which is why transformers use this one. A test changes one row and asserts every other row's output is bit-unchanged.

`gamma` starts at 1 and `beta` at 0, so the layer begins as pure normalization and learns its way out only if the loss asks.

### `eps` goes inside the square root

Outside, it would not prevent a division by zero when the variance is genuinely zero — a constant row, which happens with padding — and the gradient would still be infinite. Inside, a constant row normalizes to zeros with a finite gradient, and tests pin both.

The cost is that the output std is `0.999996`, not exactly 1, and **scale invariance is not exact**: at 7× the input, `eps` carries 49× less relative weight, shifting the output by ~2e-5. Shift invariance *is* exact, since the mean is subtracted outright. A test identifies `eps` as the cause by shrinking it and watching the gap close — the same species of finding as Adam's `eps` on Day 7.

## Residuals: what depth costs, measured

```
 depth         plain      residual         ratio
     1     5.136e-01     1.011e+00            2x
     5     3.002e-02     9.154e-01           30x
    10     2.636e-03     1.312e+00          498x
    20     7.189e-06     2.676e+00       372188x
    40     5.745e-10     8.114e+00  14121807468x
```

`d(x + f(x))/dx = 1 + f'(x)`. That **1** is a road from the loss to every earlier layer that depth cannot attenuate: a product of `(1 + small)` terms stays near 1, where a product of `small` terms does not. At 40 layers the plain stack delivers `5.7e-10` — the first layer may as well not be connected.

This is **Day 2's saturation table seen from the other end**. There, `tanh'(6) = 2.5e-05`; here forty such factors multiply together.

> Honest caveat, and a test: the residual column **grows** — 1.0 at one layer, 8.1 at forty. Residuals prevent vanishing; they do not by themselves prevent inflation. Holding that in check is part of what LayerNorm is for.

## Pre-norm versus post-norm

```
pre-norm  x + f(norm(x))   |grad at input| = 4.025e+01
post-norm norm(x + f(x))   |grad at input| = 1.412e-18
```

**Nineteen orders of magnitude**, at only twelve layers.

Pre-norm leaves the identity path completely clear — a gradient crosses the whole stack without passing through a single normalization. Post-norm, the original 2017 arrangement, puts a LayerNorm *on* that path at every layer, and each one rescales whatever passes through.

Post-norm does train; it needs a learning-rate warmup to do so, and pre-norm generally does not. This table is why. Every modern transformer uses pre-norm.

## Run it

```bash
# demo (normalization, the depth table, pre vs post-norm, constant rows)
python normalization.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_working_gpt/day13_layernorm_residual

# tests — from inside this folder
python -m unittest
python test_normalization.py

# the docstring examples are runnable too
python -m doctest normalization.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Every component now exists. **Day 14** adds the feed-forward network — where two thirds of a transformer's parameters actually live, which surprises people who assume attention dominates — and assembles the block.
