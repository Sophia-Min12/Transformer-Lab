# Day 4 · Matrix Multiplication and the Chain Rule in Two Dimensions

> Day 3 could express a linear layer as `(x * w).sum(axis=1)`. Correct, clumsy, and slow — it materialises an `(n, in, out)` array where a matrix product needs none. Every real network is mostly `@`.

## The derivative

```
C = A @ B          A is (n, k),  B is (k, m),  C is (n, m)

dL/dA = dL/dC @ Bᵀ
dL/dB = Aᵀ @ dL/dC
```

Two matrix products. No sums, no loops. It is short enough to look like a guess, so it is worth seeing where it comes from — and worth noticing that **the shapes force it**.

### Why the shapes force it

`dL/dA` has to be shaped like `A`, which is `(n, k)`. The only things available are `dL/dC` at `(n, m)` and `B` at `(k, m)`. There is exactly one way to combine those into `(n, k)`:

```
(n, m) @ (m, k)  →  (n, k)          i.e.  dC @ Bᵀ
```

Likewise `dL/dB` must be `(k, m)`, and `Aᵀ @ dC` is `(k, n) @ (n, m)` → `(k, m)`. Shape analysis alone pins both answers, which is why matmul backward is hard to get *wrong* — and the one way it is easy to get wrong is to swap the transposes, which produces correctly-shaped garbage **whenever the matrices are square**. A test constructs exactly that mistake on 3×3 inputs and shows the gradient check still catches it.

### Where it comes from

Entry-wise, `C[i,j] = Σₖ A[i,k]·B[k,j]`. So `∂C[i,j]/∂A[i,k] = B[k,j]`, and every `j` contributes:

```
dL/dA[i,k] = Σⱼ dL/dC[i,j] · B[k,j]
```

which is the `(i,k)` entry of `dC @ Bᵀ`. The matrix form is just that sum written without indices.

## Batching comes free

Only the **last two** axes take part in the product. Anything in front is a batch dimension, `np.matmul` broadcasts it, and `unbroadcast` folds it back on the way down:

```python
Tensor(np.ones((2, 3, 4))) @ Tensor(np.ones((4, 5)))   # (2, 3, 5)
```

A test checks that the shared `(4, 5)` operand's gradient is summed over the batch. This is what makes multi-head attention (Day 11) a single call instead of a loop over heads.

## Two things that raise instead of guessing

**1-D operands are rejected.** NumPy silently promotes a `(k,)` to `(1, k)` or `(k, 1)` depending on which side it sits, then demotes the result — which quietly changes what the gradient shapes mean. Asking for an explicit `reshape(1, k)` costs one call and removes a whole class of confusion.

**Mismatched inner dimensions are named:**

```
ValueError: shapes (3, 4) and (5, 2) do not line up: 4 != 5
```

## One line of NumPy interop

```python
__array_ufunc__ = None
```

Without it, `np.ones((2,3)) @ tensor` **raises** instead of falling back to `__rmatmul__`. `np.ndarray.__matmul__` tries to coerce the `Tensor` and fails rather than returning `NotImplemented`, so Python never gets the chance to try the reflected operator. Setting `__array_ufunc__ = None` makes NumPy stand back.

This is Day 4's only genuinely fiddly line, and it is the sort of thing that looks like a mystery bug the first time a raw array ends up on the left of an operator.

## What it buys

```
x @ w                         3.6 ms
(x * w).sum() over an axis  270.0 ms    (74x slower)
```

On 256×256 inputs, on this machine. And the slow version allocates a **256×256×256** array — 16 million floats — to compute something that needs none. The speedup is not the main point; the memory is. That intermediate is what makes the naive form impossible at real sizes, not merely slow.

## Level 1 closes here

`linear(x, W, b)` is one line and its gradient falls out of `@` and `+` with nothing new written:

```python
def linear(x, weight, bias):
    return x @ weight + bias
```

A test trains a linear model by repeated gradient steps and checks it recovers the true weights to within 0.05 — the first thing in this repo that actually *learns*, using only four days of machinery.

## Run it

```bash
# demo (shape analysis, gradient checks, batching, the speed comparison)
python matmul.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_autograd/day04_matmul

# tests — from inside this folder
python -m unittest
python test_matmul.py

# the docstring examples are runnable too
python -m doctest matmul.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The autograd is finished. Everything from here is built out of the five operations in it, and no new calculus is required until attention.

**Level 2** wraps the machinery in the shapes a practitioner actually handles: a `Module` base and a `Linear` layer (Day 5), a numerically stable softmax and cross-entropy (Day 6), optimizers (Day 7), and embeddings (Day 8).
