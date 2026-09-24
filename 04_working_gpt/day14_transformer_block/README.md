# Day 14 · The Feed-Forward Network and the Full Transformer Block

> Attention moves information **between positions**. The feed-forward network moves it **between dimensions**, independently at every position, with no mixing across time at all.

```
change position 3; every other position moves by 0.00e+00
```

Exactly zero — a test asserts it. That division of labour is the whole design, and it is why a block alternates the two.

## Two thirds of a transformer is the part nobody talks about

```
   dim   attention   feed-forward   norms   ff share
    64      16,384         33,088     256       67%
   256     262,144        525,568   1,024       67%
   768   2,359,296      4,722,432   3,072       67%
```

Attention costs `4d²` — four `d×d` projections. The feed-forward, at the conventional `4d` hidden width, costs `8d²`. **The ratio is exactly 2:1 at every width**, and tests pin both the ratio and the 67% share.

The LayerNorms are under 0.1% of the total, which is worth knowing before optimizing them.

## GELU

```
     x      relu      gelu
  -3.0    0.0000   -0.0036
  -1.0    0.0000   -0.1588
   0.5    0.5000    0.3457
   3.0    3.0000    2.9964

relu at x=-1: gradient +0.0000
gelu at x=-1: gradient -0.0830
```

ReLU's hard zero switches a unit **off**: no output, and no gradient, so nothing pulls it back. GELU keeps a little of the negative side, so a unit that is slightly wrong still receives signal.

The `tanh` approximation implemented here is what the GPT-2 code shipped; a test checks it against the exact `0.5x(1 + erf(x/√2))` form to three decimals.

## The block

```python
x = x + attention(norm(x))
x = x + feedforward(norm(x))
```

Two sub-layers, two normalizations, two identity paths, and **nothing new**. Stacking this is the entire architecture — GPT-2 is twelve, GPT-3 is ninety-six.

The order is not arbitrary: attention gathers information from other positions, and the feed-forward then processes what was gathered. Reversing them would spend the larger layer on information the position already had.

Two properties are tested on the assembled block rather than assumed:

**Causality survives a stack.** Each block is causal, so a three-block composition must be — checked with Day 10's gradient probe on the *stack*, not on one layer.

**The gradient survives depth.** Twelve blocks, `|grad at input| = 8.2`. Day 13's residuals doing their job inside something real.

## Run it

```bash
# demo (position independence, the parameter split, GELU, the block)
python block.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_working_gpt/day14_transformer_block

# tests — from inside this folder
python -m unittest
python test_block.py

# the docstring examples are runnable too
python -m doctest block.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Every component exists and every one has been checked against finite differences. **Day 15** wires them into a character-level GPT — embeddings in, blocks, a final norm, a projection back to vocabulary — and trains it until it produces text.
