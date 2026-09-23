# Day 1 · Scalar Autograd

> A neural network is a very large composite function, and training it means knowing how the loss changes when each of a million parameters is nudged. By hand is out of the question. Numerically is far too slow. **Reverse-mode automatic differentiation** is the third option, and it costs roughly one extra pass over the same computation.

## The idea is smaller than its name

Every arithmetic operation knows its own local derivative. For `c = a * b`, `dc/da` is `b` — no calculus required beyond that one fact per operation.

So: let every value remember **the operation that produced it** and **which values fed it**. Then the chain rule can be applied backwards from the result to every input, multiplying local derivatives along the way.

That is the whole algorithm. This day builds it for single numbers; Day 3 swaps in arrays and almost nothing changes, which is the point — **autograd is not about tensors.**

## The `Value` class

Three fields do the work:

| Field | What it is |
|---|---|
| `data` | the number |
| `grad` | ∂(whatever you called `backward()` on) / ∂(this value) |
| `_backward` | a closure that pushes this value's gradient one step back to its inputs |

```python
a = Value(2.0)
b = Value(3.0)
c = a * b
c.backward()
a.grad, b.grad     # (3.0, 2.0)
```

## Three decisions worth defending

**`+=`, never `=`, inside `_backward`.** A value used twice receives a contribution from *each* use:

```python
x = Value(3.0)
(x + x).backward()
x.grad     # 2.0 — with `=` it would be 1.0
```

This is the single most common way a hand-written autograd is quietly wrong. It produces gradients that are too small rather than obviously broken, so the model still trains — just worse than it should. A test pins it.

**Reverse *topological* order.** A node must not propagate its gradient until **every** consumer has contributed to it. A topological sort, reversed, guarantees that. Any other traversal order silently under-counts, with the same hard-to-spot symptom.

**`Value ** Value` raises.** The derivative with respect to an exponent needs `a**b · ln(a)`, which is not implemented here. Raising beats computing the wrong number silently — the same principle as Day 6's IDF schemes over in NLP-Lab.

**`backward()` accumulates across calls.** Calling it twice without `zero_grad()` doubles the gradients. That is not a defect to patch; it is exactly the behaviour that makes gradient accumulation across mini-batches possible, and every framework works this way. A test documents it rather than forbidding it.

## The finite-difference check

This is the habit the entire repo is built on.

```
df/dx  ≈  (f(x + h) − f(x − h)) / 2h
```

`check_gradient(f, inputs)` runs autograd and the numerical approximation side by side and raises if they disagree. Every derivative introduced from here on gets one.

**Central, not forward.** `(f(x+h) − f(x−h)) / 2h` has error proportional to `h²`; the forward difference `(f(x+h) − f(x)) / h` is only `h`. That buys several digits of agreement for free — and those digits are the difference between a meaningful check and a decorative one. A test makes the point at `x = 0` for `f(x) = x²`, where the central difference is exactly 0 and a forward difference would report `h`.

`h ≈ 1e-5` is the usual compromise: smaller and the subtraction loses precision to floating point, larger and the approximation drifts. **Day 2 measures that trade-off** rather than asserting it.

### The check can actually fail

A gradient checker nobody has seen fail is not evidence of anything. One test subclasses `Value`, deliberately breaks the product rule by dropping a factor, and asserts the check catches it:

```python
def _backward():
    self.grad += out.grad            # missing `* other.data`
    other.grad += self.data * out.grad
```

## Why this matters more than it looks

The last test in the file is the entire premise of training, on the smallest possible model:

```python
x = Value(5.0)
loss = (x - 3.0) ** 2
loss.backward()
stepped = Value(x.data - 0.1 * x.grad)
assert ((stepped - 3.0) ** 2).data < loss.data
```

Take a step against the gradient, and the loss goes down. Everything in Levels 2–4 is that line with more parameters.

## Run it

```bash
# demo (a graph differentiated, gradient checks, topological order)
python autograd.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_autograd/day01_scalar_autograd

# tests — from inside this folder
python -m unittest
python test_autograd.py

# the docstring examples are runnable too
python -m doctest autograd.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

**Day 2** adds the operations a network actually needs — `tanh`, `exp`, `log` — and measures how the finite-difference step size `h` trades floating-point noise against approximation error. **Day 3** replaces the scalar with a NumPy array, which turns out to change the `_backward` closures and nothing else.
