# Day 5 · Linear Layers, Initialization, and a `Module` Base

> Level 1 built an autograd. Everything from here is that autograd wearing the shapes a practitioner actually handles.

## A `Module` answers one question

*Which tensors do you want trained?* The optimizer on Day 7 needs that list and **nothing else** about the model — not the architecture, not the forward pass, just an array of tensors with gradients.

```python
net = Sequential(Linear(8, 16), Tanh(), Linear(16, 4))
net.parameters()        # 4 tensors, 212 numbers
```

The walk recurses through attributes, lists and nested modules. Two details it gets right:

**A parameter the walk misses never trains.** It produces no error — the model just plateaus, and nothing says why. `ModuleList` exists so that layers held in a collection stay discoverable.

**A shared parameter is listed once.** Weight tying (Day 15 ties the embedding to the output projection) puts the same tensor in two places; listing it twice would apply every update twice. The walk deduplicates by identity, and a test pins it.

## Initialization is not a detail

The weights are random, but the **scale** of that randomness is chosen to keep the variance of activations roughly constant with depth. Thirty tanh layers, width 128, standard deviation of the activations as they go down:

```
init         std @1      @10      @20      @30
too small    0.0984   0.0000   0.0000   0.0000
lecun        0.6231   0.2303   0.1527   0.1326
xavier       0.6300   0.2310   0.1526   0.1284
too large    0.9305   0.9275   0.9233   0.9228
```

**Too small and the signal is gone by layer 10** — the network's output does not depend on its input in any numerically detectable way, so there is nothing to learn.

**Too large and tanh saturates.** The std sits near 1 because every unit is pinned at ±1, and Day 2 already measured what that does: `tanh'(±6) ≈ 2.5e-5`. The forward pass looks healthy and the gradient is dead.

| Mode | Scale | For |
|---|---|---|
| `xavier` | `√(2/(fan_in+fan_out))` | balances forward and backward — the default for tanh |
| `he` | `√(2/fan_in)` | ReLU zeroes half its inputs, so it needs twice the variance |
| `lecun` | `√(1/fan_in)` | preserves forward variance only |

A test asserts `he² = 2 · lecun²`, which is the ReLU correction stated as an identity rather than a folk rule.

## Zero weights are the one initialization that cannot work

```
every output unit receives the same gradient: True
```

Identical weights get identical gradients, so they stay identical for ever — a layer of 100 units would learn exactly one thing, permanently. A test pins the symmetry, and a companion test confirms random initialization breaks it.

**The bias, by contrast, starts at zero.** It has no fan-in to balance, and a random bias only adds noise to the first few steps.

## Shape conventions

`W` is `(in_features, out_features)`, so a batch of rows stays a batch of rows and **no transpose appears in the forward pass**:

```
(batch, in) @ (in, out) -> (batch, out)
```

Extra leading dimensions pass straight through, because Day 4's matmul treats everything before the last two axes as batch. `(2, 6, 4) -> (2, 6, 3)` works without a special case — which is exactly the shape attention will hand it on Day 9.

## Run it

```bash
# demo (parameter walk, the depth table, symmetry, gradient checks)
python modules.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_network_pieces/day05_linear_and_modules

# tests — from inside this folder
python -m unittest
python test_modules.py

# the docstring examples are runnable too
python -m doctest modules.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The network can produce numbers; it cannot yet be told it is wrong. **Day 6** adds softmax and cross-entropy, where the whole difficulty is numerical: the obvious implementation of both overflows, and the standard fix looks like a trick until you have seen it fail.
