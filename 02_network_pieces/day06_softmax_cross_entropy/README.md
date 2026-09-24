# Day 6 · Softmax and Cross-Entropy, Numerically Stable

> A classifier turns scores into probabilities and then measures how wrong they are. Both steps are one line of mathematics, and **both of those lines fail on real numbers**.

## Failure one: `exp` overflows

```
max logit      10   naive: [0.9998, 0.0, 0.0001]    stable: [0.9998, 0.0, 0.0001]
max logit     100   naive: [1.0, 0.0, 0.0]          stable: [1.0, 0.0, 0.0]
max logit     710   naive: nan / inf                stable: [1.0, 0.0, 0.0]
max logit    1000   naive: nan / inf                stable: [1.0, 0.0, 0.0]
```

`exp(710)` is `inf` in float64. `inf/inf` is `nan`. One `nan` reaches every parameter in a single backward pass and the run is over — with no indication of where it started.

A logit of 710 is not exotic. An unnormalized model a few hundred steps into training produces them routinely, which is why `naive_softmax` is kept in the file: so the overflow can be *demonstrated* rather than described. A test pins that it starts failing between 700 and 800.

### The fix is exact, not approximate

**`softmax(x) = softmax(x − c)` for any constant `c`.** The shift cancels between numerator and denominator. Choosing `c = max(x)` makes the largest exponent exactly `exp(0) = 1`, so nothing can overflow and the smallest terms underflow harmlessly to zero.

```
softmax(x)       [0.138467, 0.063888, 0.648529, 0.130079, 0.019038]
softmax(x - 100) [0.138467, 0.063888, 0.648529, 0.130079, 0.019038]
identical: True
```

The shift is deliberately computed from `x.data` and wrapped as a **constant**, outside the graph. That is not laziness: the output genuinely does not depend on `c`, so `∂output/∂c = 0`, and routing a gradient through the max would add a term that must cancel to nothing anyway.

## Failure two: `log` of something that underflowed

```
log(softmax(x))  [[0.0, -inf]]
log_softmax(x)   [[0.0, -800.0]]
```

A confident model assigns probabilities like `1e-320`. A little more confidence and that rounds to exactly `0`, and `log(0)` is `−inf`.

`log_softmax` computes `x − c − log(Σ exp(x − c))` and **never forms the probability at all**. `−800` is an ordinary float; it only becomes `−inf` if you take a detour through probability space and come back.

This is why `cross_entropy` takes **logits, not probabilities**. Every serious implementation does. Handing it a softmax output would force it to take the log of something already collapsed to zero, and fusing the two steps is precisely what keeps the whole thing finite.

## The gradient is `(p − y) / n`

```
autograd max |difference| from (p - y)/n: 0.00e+00
```

The softmax derivative is a full `(classes, classes)` Jacobian and the log derivative is a reciprocal. Composing them cancels almost everything, and what survives is **"predicted probability minus the truth"**, scaled by the batch size.

That cancellation is the reason softmax and cross-entropy are always fused: the fused backward pass is one subtraction, where the honest chain through both would build and multiply a Jacobian per example.

Two tests make the structure visible:
- **Each row's gradient sums to zero**, because `p` sums to 1 and `y` sums to 1. Shifting all logits by a constant cannot change the loss, and the gradient knows it.
- **The correct class gets a negative gradient** and the others positive — push the right logit up, the rest down.

## The number to compare an untrained model against

```
uniform over 2 classes                0.6931
uniform over 10 classes               2.3026
confident and right                   0.0000
confident and wrong                  10.0000
```

Uniform over `k` classes is **exactly `log(k)`**. A freshly initialized classifier should land there, and a test asserts it for `k` in 2, 5, 10 and 100. If your first loss is not near `log(k)`, something is wrong before training has begun — which is the cheapest bug-catch in the whole pipeline.

## Run it

```bash
# demo (both overflows, shift invariance, the fused gradient)
python losses.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_network_pieces/day06_softmax_cross_entropy

# tests — from inside this folder
python -m unittest
python test_losses.py

# the docstring examples are runnable too
python -m doctest losses.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

There is now a model, a loss, and a gradient for every parameter. **Day 7** adds the thing that uses them — SGD and Adam — and trains something end to end for the first time, which finally makes "does it learn?" a question with an answer.
