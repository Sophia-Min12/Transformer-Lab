# Day 3 · Tensor Autograd

> A scalar autograd is correct and unusably slow. One attention head does millions of multiplications, and Python cannot run millions of `Value` objects per forward pass.

Replacing the scalar with a NumPy array changes remarkably little. **The graph, the topological sort and `backward()` are untouched** — the chain rule does not care about shapes. Only the `_backward` closures change, and almost all of them by one line.

That one line is the whole difficulty.

## Broadcasting duplicates data, so the gradient must sum

```
(2,3) + (3,)  ->  (2,3)

d/d rows  shape (2, 3):  [1. 1. 1.]   every entry used once
d/d bias  shape (3,):    [2. 2. 2.]   each entry used twice
```

The bias gradient is **2, not 1**. Broadcasting used each bias entry twice in the forward pass, so two gradient contributions come back — which is exactly Day 1's `+=` rule, restated in shapes.

`unbroadcast(gradient, shape)` does the summing, in two steps:

1. Broadcasting **prepends** axes, so leading dimensions the original did not have are summed away entirely.
2. Within the remaining axes, a size-1 dimension was stretched, so it is summed with `keepdims` to stay size 1.

### Why this bug is dangerous rather than merely wrong

Forget the `unbroadcast` and the gradient comes back with the wrong *shape* — and NumPy will very often broadcast that wrong shape into something plausible instead of raising. The parameter update then happens, the loss still goes down, and the model simply trains worse than it should for reasons nothing reports.

A test subclasses `Tensor`, deletes the `unbroadcast` call from `__add__`, and asserts the resulting gradient has the wrong shape. Another test checks that unbroadcasting **conserves the total** — summing cannot invent or lose gradient mass.

## Sum and broadcast are adjoint

Worth seeing once, because it makes `unbroadcast` obvious instead of fiddly:

- The derivative of a **sum** is to *copy* the incoming gradient to every element that fed it.
- The derivative of a **broadcast** is to *sum* the incoming gradient over the stretched axes.

They are the same operation run in opposite directions. `Tensor.sum()`'s backward is a `broadcast_to`; `__add__`'s backward is a sum.

## `backward()` refuses a non-scalar output

```
ValueError: backward() needs a scalar output, got shape (2,);
reduce with .sum() or .mean() first
```

A non-scalar has no single derivative to propagate — you would have to say which combination of its entries you meant. Seeding with ones (the obvious implementation) silently means "the sum of the entries", which is a guess. Losses are scalars; call `.sum()` or `.mean()` and mean it.

## Operations added

| Operation | Gradient |
|---|---|
| `+`, `*`, `**`, `-`, `/` | as Day 1, plus `unbroadcast` |
| `exp`, `log`, `tanh`, `relu` | element-wise, as Day 2 |
| `sum(axis, keepdims)` | broadcast the gradient back |
| `mean(axis)` | a sum scaled by `1/n` |
| `reshape`, `transpose` / `.T` | the reverse reshape or transpose — no arithmetic |

`reshape` and `transpose` are worth a glance precisely because their backward passes contain **no arithmetic at all**. A shape change moves numbers around; its derivative moves them back.

## Why reverse mode exists at all

`numerical_gradient` perturbs **one element at a time**, so it costs two forward passes per parameter. Two tests count the calls:

```
a function of 10 parameters:
  finite differences: 20 forward passes
  backprop:            1 forward pass
```

At 200 parameters that is 400 passes versus 1. A GPT has hundreds of millions. This is the entire economic argument for backpropagation, and it is why the finite-difference check is a *test* and never a training method.

## Run it

```bash
# demo (broadcasting both directions, adjointness, gradient checks)
python tensor.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_autograd/day03_tensor_autograd

# tests — from inside this folder
python -m unittest
python test_tensor.py

# the docstring examples are runnable too
python -m doctest tensor.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

One operation is still missing, and it is the one a network is mostly made of: **matrix multiplication**. `(x * w).sum(axis=1)` is standing in for it here and is both slower and clumsier than `x @ w`. **Day 4** adds `@`, whose backward pass is two more matrix multiplications and the first derivative in this repo that is genuinely worth deriving on paper.
