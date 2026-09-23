# Day 2 · More Operations, and What the Gradient Check Costs

> Day 1's `Value` can add, multiply and raise to a constant power — enough for polynomials, not enough for a network. Every activation needs `exp`, `tanh` or a comparison; every loss needs `log`.

Each new operation is the same three lines: compute the value, record the inputs, write the local derivative. So the first half of this day is about *which* derivatives. The second half stops trusting the finite-difference check and measures it.

## The new operations

| Operation | Derivative | Note |
|---|---|---|
| `exp()` | `e^x` | the derivative **is** the output |
| `log()` | `1/x` | raises on `x ≤ 0` |
| `tanh()` | `1 − tanh²x` | written from the output too |
| `sigmoid()` | `σ(1−σ)` | overflow-safe for large negative `x` |
| `relu()` | `1` if `x>0` else `0` | undefined at 0 — see below |

Notice how many are written in terms of the **output** rather than the input. That is not a coincidence: it means the backward pass reuses a number the forward pass already computed, which is why frameworks cache activations.

`sigmoid` uses `exp(x)/(1+exp(x))` for negative input rather than `1/(1+exp(-x))`. The textbook form raises `OverflowError` at `x = -800`; a test pins that this one does not.

`log` raises on non-positive input rather than returning `-inf` or `nan`. **A `nan` in the loss reaches every parameter in a single backward pass and gives no clue where it started.** An exception names the spot.

## ReLU is not differentiable at zero

The left derivative is 0, the right derivative is 1, and there is no single answer. Every framework picks one anyway; this picks 0, matching PyTorch.

What is worth seeing is that the gradient check **disagrees**, and correctly:

```
autograd says 0.0                                  (PyTorch's choice)
a finite difference straddling the kink says 0.5   (the average of both sides)
```

A test asserts that `check_gradient` *fails* at exactly `x = 0` for ReLU. Neither number is wrong — the derivative does not exist there — and pretending otherwise by loosening the tolerance would hide a real fact about the function.

## Saturation is the vanishing gradient, in one line

```
tanh'( 0.0) = 1.000e+00
tanh'( 1.0) = 4.200e-01
tanh'( 3.0) = 9.866e-03
tanh'( 6.0) = 2.458e-05
tanh'(10.0) = 8.245e-09
```

A deep stack multiplies numbers like these together. A test builds ten saturating layers and shows the input gradient has essentially vanished — which is the entire reason Level 4 needs residual connections, and the reason ReLU replaced tanh in most architectures.

## The step size is a trade-off, not a constant

This is the measurement the day exists for. Relative error of the central difference for `f(x) = exp(x³)` at `x = 0.6`:

```
       h    relative error   cause
   1e-01          2.88e-02   truncation
   1e-02          2.92e-04   truncation
   1e-03          2.92e-06   truncation
   1e-04          2.92e-08   truncation
   1e-05          2.86e-10   truncation
   1e-06          7.89e-11   <- best
   1e-07          5.84e-10   cancellation
   1e-08          5.21e-09   cancellation
   1e-10          4.19e-07   cancellation
   1e-12          1.37e-05   cancellation
   1e-14          6.06e-03   cancellation
```

**A clean U.** The two halves have completely different causes:

- **Large `h` — truncation.** The difference quotient only approximates the derivative. Error falls as `h²`, and the table shows it: dividing `h` by 10 divides the error by ~100.
- **Small `h` — cancellation.** `f(x+h)` and `f(x−h)` agree to more and more digits, so subtracting them throws away most of the significant ones, and dividing by a tiny `h` magnifies what survives. Error rises as `1/h`.

The minimum sits near `ε^(1/3) ≈ 6e-6` for double precision, which is exactly why `1e-5` is the conventional default rather than "as small as possible".

Three tests pin this: that the error is **not monotonic** (so "smaller `h` is better" is false), that the best step is in the interior, and that the large-`h` side really does fall quadratically.

### The check can be wrong when the gradient is right

```python
check_gradient(lambda a: (a * a * a).exp(), [0.6], h=1e-5)   # passes
check_gradient(lambda a: (a * a * a).exp(), [0.6], h=1e-14)  # FAILS
```

Same derivative, same code, different step size. Worth knowing before spending an afternoon debugging a derivation that was fine — when a gradient check fails, the check is a suspect too.

## Relative error, not absolute

`relative_error` divides by the larger magnitude. A gradient of `1e6` and one of `1e-6` cannot share an absolute threshold, and `max(1, ...)` keeps the quotient finite when both are near zero. Every tolerance from here on is relative.

## Run it

```bash
# demo (gradient checks, the U-curve, saturation, the ReLU kink)
python operations.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_autograd/day02_operations

# tests — from inside this folder
python -m unittest
python test_operations.py

# the docstring examples are runnable too
python -m doctest operations.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

`Value` can now express a whole neuron, and a network of them would work — one scalar at a time, slowly. **Day 3** replaces the scalar with a NumPy array. The interesting part is how little changes: the graph, the topological sort and `backward()` are untouched, and only the `_backward` closures learn about shapes.
