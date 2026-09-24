# Day 12 · Positional Encoding

> Day 11's second failed attempt ran into this without naming it. **Attention is permutation-equivariant**: shuffle the input positions and the outputs come back shuffled the same way, otherwise unchanged.

```
shuffle the input, un-shuffle the output, compare:
  max difference = 1.67e-16
```

Zero, to floating-point noise. The layer genuinely cannot tell one ordering from another, so *"dog bites man"* and *"man bites dog"* produce the same set of outputs and no amount of training separates them. That disqualifies it as a language model.

`permutation_gap()` is the probe: run the layer on `x` and on `x[order]`, un-shuffle the second, compare. It is the measurement Day 11 needed and did not have.

## What I got wrong about causal masking

My first draft said masking does not help, because it only forbids looking ahead. **The measurement disagreed:**

```
causal layer, same probe: 9.89e-01
```

Not zero. Causal masking **does** break permutation equivariance — position `t` sees exactly the set `{0..t}`, and shuffling changes which tokens fall inside that set, so the computation genuinely differs. Decoder-only models can and do learn position from the mask alone; this is a real result, not a curiosity.

Explicit encodings remain standard anyway, because the mask's signal is weak and indirect: it tells a position **how many** tokens precede it, never *which* ones or *how far away* they are. A test pins both facts — plain attention at `< 1e-12`, causal at `> 1e-3`.

## The sinusoids

```
pos       0       1       2       3       4       5       6       7
  0   0.000   1.000   0.000   1.000   0.000   1.000   0.000   1.000
  1   0.841   0.540   0.100   0.995   0.010   1.000   0.001   1.000
  2   0.909  -0.416   0.199   0.980   0.020   1.000   0.002   1.000
```

Each **pair** of dimensions is a clock hand, and the hands run at geometrically spaced rates — the leftmost spins every few positions, the rightmost barely moves across the whole sequence. Reading all of them identifies a position the way reading every hand of a clock identifies a time.

Three properties, each tested:

**Equal norm everywhere.** `sin² + cos² = 1` per pair, so every position has norm `√(dim/2)` exactly — `5.6569` for dim 64. No position shouts louder than another, which matters because this vector is *added* to the token embedding.

**Distance grows, then plateaus.**

```
k=   1: 1.4718      k=  64: 6.0156
k=   2: 2.7189      k= 256: 6.4315
k=   8: 4.3786
```

Nearby positions are separated finely, distant ones only coarsely — which is the resolution a language model actually needs.

**Relative position is a fixed rotation.**

```
rotation from position  5 to  8: +3.000000 rad
rotation from position 20 to 23: +3.000000 rad
rotation from position 40 to 43: +3.000000 rad
```

`PE(pos + k)` is the *same* rotation of `PE(pos)` at every `pos`, so "three positions back" is one thing to learn rather than a separate thing at each location.

> That table initially read `+3.000`, `−3.283`, `−3.283`. The two values differ by exactly `2π` and are the same rotation — `atan2` returns `(−π, π]`, so a difference of angles can come back a full turn out. Folding it back with `atan2(sin θ, cos θ)` fixes it. Worth flagging because an unwrapped angle comparison looks like a real disagreement and is not, and the same fold appears in PhysicalAI-Lab Day 1.

## Sinusoidal versus learned

```
learned table: 32 parameters, max length 4
sinusoidal:     0 parameters, no maximum

asking for 5 positions: IndexError: sequence of 5 exceeds max_length 4
sinusoidal at 5 positions: (5, 8)
sinusoidal at 5000:        (5000, 8)
```

The learned table is strictly **more expressive** and strictly **less general**. GPT-2 uses learned; the original paper used sinusoids and reported nearly identical results.

The failure mode is the deciding factor: a learned table raises `IndexError` past its length, which is far better than returning a confident wrong answer. Extending a trained model's context means adding rows that have never been trained.

## Run it

```bash
# demo (the equivariance probe, the sinusoids, the two encodings)
python positional.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_attention/day12_positional_encoding

# tests — from inside this folder
python -m unittest
python test_positional.py

# the docstring examples are runnable too
python -m doctest positional.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Level 3 closes here

Attention now looks only backwards, runs several relationships in parallel, and knows where it is. **Level 4** adds the two pieces that make a stack of these trainable — LayerNorm and residual connections — assembles the transformer block, and trains a character-level GPT.
