"""Day 2 — More operations, and what the gradient check actually costs.

Day 1's ``Value`` can add, multiply and raise to a constant power. That is
enough for polynomials and not enough for a network: every activation
function needs ``exp``, ``tanh`` or a comparison, and every loss needs
``log``. Each one is the same three lines — compute the value, record the
inputs, write the local derivative — so this day is mostly about *which*
derivatives, and about one thing that is easy to get wrong.

The second half measures the finite-difference check rather than trusting
it. ``h`` too large and the approximation drifts; ``h`` too small and
catastrophic cancellation eats the answer. There is a sweet spot, it is
narrower than people expect, and it is worth seeing the U-curve once.
"""

from __future__ import annotations

import math


# reused from day01
class Value:
    """A scalar that records how it was computed.

    >>> (Value(2.0) * Value(3.0)).data
    6.0
    """

    def __init__(self, data: float, children: tuple = (), operation: str = "") -> None:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise TypeError(f"Value takes a number, got {type(data).__name__}")
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._previous = set(children)
        self._operation = operation

    def __repr__(self) -> str:
        return f"Value(data={self.data:g}, grad={self.grad:g})"

    @staticmethod
    def _wrap(other) -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other) -> "Value":
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent) -> "Value":
        if isinstance(exponent, Value):
            raise TypeError("exponent must be a constant, not a Value")
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
            raise TypeError(f"exponent must be a number, got {type(exponent).__name__}")
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-self._wrap(other))

    def __truediv__(self, other) -> "Value":
        return self * (self._wrap(other) ** -1)

    def __radd__(self, other) -> "Value":
        return self + other

    def __rmul__(self, other) -> "Value":
        return self * other

    def __rsub__(self, other) -> "Value":
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other) -> "Value":
        return self._wrap(other) * (self ** -1)

    # ---- new on day 2 -------------------------------------------------

    def exp(self) -> "Value":
        """``d(e^x)/dx = e^x`` — the derivative *is* the output.

        Worth noticing because it means the backward pass can reuse
        ``out.data`` instead of recomputing anything. Most activation
        derivatives have this shape.

        >>> round(Value(0.0).exp().data, 6)
        1.0
        """
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward():
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        """Natural log. ``d(ln x)/dx = 1/x``.

        Raises on non-positive input rather than returning ``-inf`` or
        ``nan``. A ``nan`` in a loss propagates to every parameter in one
        backward pass and gives no clue where it came from; an exception
        names the spot.

        >>> round(Value(math.e).log().data, 6)
        1.0
        """
        if self.data <= 0:
            raise ValueError(f"log of a non-positive number: {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        """``d(tanh x)/dx = 1 - tanh²x`` — again, written from the output.

        ``math.tanh`` saturates gracefully at large ``|x|``, so no
        overflow guard is needed; the gradient simply goes to zero, which
        is the vanishing-gradient problem in one line.

        >>> Value(0.0).tanh().data
        0.0
        """
        value = math.tanh(self.data)
        out = Value(value, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - value * value) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        """``d(σ x)/dx = σx · (1 - σx)``.

        Computed in the branchless-but-stable form: for negative input,
        ``exp(x) / (1 + exp(x))`` instead of ``1 / (1 + exp(-x))``, since
        the latter overflows for large negative ``x``.

        >>> Value(0.0).sigmoid().data
        0.5
        """
        if self.data >= 0:
            value = 1.0 / (1.0 + math.exp(-self.data))
        else:
            scaled = math.exp(self.data)
            value = scaled / (1.0 + scaled)
        out = Value(value, (self,), "sigmoid")

        def _backward():
            self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """``max(0, x)``, with the derivative at exactly 0 defined as 0.

        ReLU is **not differentiable at zero** — the left derivative is 0
        and the right derivative is 1. Every framework picks one anyway;
        this picks 0, matching PyTorch. The choice is arbitrary and the
        set of inputs where it matters has measure zero, but a finite
        difference straddling the kink will disagree, which is why the
        gradient check skips that point rather than pretending.

        >>> Value(-2.0).relu().data
        0.0
        """
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ---- graph walk (reused from day01) -------------------------------

    def topological_order(self) -> list["Value"]:
        ordered: list[Value] = []
        seen: set[int] = set()

        def visit(node: "Value") -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._previous:
                visit(child)
            ordered.append(node)

        visit(self)
        return ordered

    def backward(self) -> None:
        ordered = self.topological_order()
        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()

    def zero_grad(self) -> None:
        for node in self.topological_order():
            node.grad = 0.0


# reused from day01, extended to functions written against the Value API
def numerical_gradient(function, inputs: list[float], index: int, h: float = 1e-5) -> float:
    """Estimate d function / d inputs[index] by central differences.

    Day 1's version passed raw floats, which worked only because its test
    functions used arithmetic operators that floats also have. ``a.exp()``
    does not exist on a float, so the inputs are wrapped in ``Value`` here
    and the result unwrapped — the function under test can then be written
    once and used by both halves of the check.
    """
    if not 0 <= index < len(inputs):
        raise IndexError(f"no input {index} among {len(inputs)}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    def evaluate(point: list[float]) -> float:
        result = function(*[Value(x) for x in point])
        return result.data if isinstance(result, Value) else float(result)

    forward = list(inputs)
    backward = list(inputs)
    forward[index] += h
    backward[index] -= h
    return (evaluate(forward) - evaluate(backward)) / (2 * h)


def relative_error(analytic: float, numeric: float) -> float:
    """Scale-free disagreement between two gradients.

    Absolute difference is useless for comparison: a gradient of
    ``1e6`` and one of ``1e-6`` cannot share a threshold. Dividing by the
    larger magnitude gives a number that means the same thing everywhere,
    and ``max(1, ...)`` keeps it finite when both are near zero.

    >>> relative_error(1.0, 1.0)
    0.0
    """
    return abs(analytic - numeric) / max(1.0, abs(analytic), abs(numeric))


# reused from day01, now reporting relative error
def check_gradient(function, inputs: list[float], tolerance: float = 1e-7, h: float = 1e-5):
    """Compare autograd against finite differences; raise on disagreement."""
    values = [Value(x) for x in inputs]
    output = function(*values)
    if not isinstance(output, Value):
        raise TypeError("function must return a Value")
    output.backward()

    analytic = [v.grad for v in values]
    for i, gradient in enumerate(analytic):
        numeric = numerical_gradient(function, inputs, i, h)
        error = relative_error(gradient, numeric)
        if error > tolerance:
            raise AssertionError(
                f"gradient {i} disagrees by {error:.2e}: autograd {gradient:.10f} "
                f"vs finite difference {numeric:.10f}"
            )
    return analytic


def step_size_sweep(function, inputs: list[float], index: int = 0,
                    steps=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
                           1e-9, 1e-10, 1e-12, 1e-14)):
    """Relative error of the finite difference, as ``h`` shrinks.

    Returns ``[(h, error), ...]``. The curve is U-shaped and the two
    halves have completely different causes:

    - **Large h** — truncation. The difference quotient is only an
      approximation of the derivative, and the error falls as ``h²``.
    - **Small h** — cancellation. ``f(x+h)`` and ``f(x-h)`` agree to more
      and more digits, so subtracting them discards most of the
      significant ones, and dividing by a tiny ``h`` magnifies what is
      left. The error rises as ``1/h``.

    The minimum sits near ``h = ε^(1/3) ≈ 6e-6`` for double precision,
    which is why ``1e-5`` is the conventional default.
    """
    values = [Value(x) for x in inputs]
    output = function(*values)
    output.backward()
    analytic = values[index].grad
    return [
        (h, relative_error(analytic, numerical_gradient(function, inputs, index, h)))
        for h in steps
    ]


if __name__ == "__main__":
    print("the new operations, each checked against finite differences:")
    for name, function, inputs in (
        ("exp", lambda a: a.exp(), [0.7]),
        ("log", lambda a: a.log(), [2.5]),
        ("tanh", lambda a: a.tanh(), [0.4]),
        ("sigmoid", lambda a: a.sigmoid(), [-1.2]),
        ("relu (x>0)", lambda a: a.relu(), [1.3]),
        ("relu (x<0)", lambda a: a.relu(), [-1.3]),
        ("a neuron", lambda w, x, b: (w * x + b).tanh(), [0.5, 2.0, -1.0]),
        ("log-sum-exp", lambda a, b: (a.exp() + b.exp()).log(), [1.0, 2.0]),
        ("cross-entropy-ish", lambda a: -(a.sigmoid().log()), [0.3]),
    ):
        gradients = check_gradient(function, inputs)
        print(f"  {name:<18} grad = {[round(g, 8) for g in gradients]}  ok")

    print("\nthe finite-difference step size is a trade-off, not a constant:")
    sweep = step_size_sweep(lambda a: (a * a * a).exp(), [0.6])
    best_h = min(sweep, key=lambda row: row[1])[0]
    print(f"  {'h':>8}{'relative error':>18}   cause")
    for h, error in sweep:
        if h == best_h:
            cause = "<- best"
        elif h > best_h:
            cause = "truncation"
        else:
            cause = "cancellation"
        print(f"  {h:>8.0e}{error:>18.2e}   {cause}")
    print("  large h: the quotient is only an approximation, error falls as h^2")
    print("  small h: f(x+h) and f(x-h) agree to too many digits, and subtracting")
    print("           them throws the significant ones away. error rises as 1/h")
    print("  the minimum sits near eps^(1/3) = 6e-6, which is why 1e-5 is standard")

    print("\nsaturation is the vanishing gradient, in one line:")
    for x in (0.0, 1.0, 3.0, 6.0, 10.0):
        node = Value(x)
        node.tanh().backward()
        print(f"  tanh'({x:>4.1f}) = {node.grad:.3e}")
    print("  a deep stack of these multiplies such numbers together")

    print("\nrelu is not differentiable at 0, and every framework picks anyway:")
    node = Value(0.0)
    node.relu().backward()
    numeric = numerical_gradient(lambda a: a.relu(), [0.0], 0)
    print(f"  autograd says {node.grad}  (matching PyTorch's choice)")
    print(f"  a finite difference straddling the kink says {numeric}")
    print("  neither is wrong; the derivative does not exist there")

    print("\nlog of a non-positive number raises rather than returning nan:")
    try:
        Value(-1.0).log()
    except ValueError as error:
        print(f"  ValueError: {error}")
    print("  a nan in the loss reaches every parameter in one backward pass")
    print("  and gives no clue where it started")
