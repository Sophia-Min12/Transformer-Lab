"""Day 1 — Scalar autograd: a Value that remembers how it was made.

A neural network is a very large composite function, and training it means
knowing how the loss changes when each of a million parameters is nudged.
Computing those derivatives by hand is out of the question; computing them
numerically is far too slow. **Reverse-mode automatic differentiation** is
the third option, and it costs roughly one extra pass over the same
computation.

The idea is smaller than its name. Every arithmetic operation knows its own
local derivative — for ``c = a * b``, ``dc/da`` is ``b``. If each value
remembers the operation that produced it and which values fed it, then the
chain rule can be applied backwards from the result to every input.

This day builds that for single numbers. Day 3 swaps in arrays and almost
nothing else changes, which is the point: autograd is not about tensors.
"""

from __future__ import annotations


class Value:
    """A scalar that records how it was computed.

    Three fields do the work: ``data`` is the number, ``grad`` is the
    derivative of whatever you called ``backward()`` on with respect to
    this value, and ``_backward`` is a closure that pushes this value's
    gradient one step further back to its inputs.

    >>> a = Value(2.0)
    >>> b = Value(3.0)
    >>> c = a * b
    >>> c.data
    6.0
    >>> c.backward()
    >>> a.grad, b.grad
    (3.0, 2.0)
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
        """``d(a + b)/da = 1``, so addition passes its gradient through."""
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # += rather than =, because a value used twice receives a
            # contribution from each use. See test_reused_value.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        """``d(a * b)/da = b`` — each factor is scaled by the other."""
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent) -> "Value":
        """``d(a**n)/da = n * a**(n-1)``, for a constant exponent.

        A ``Value`` exponent would need ``a**b * ln(a)`` as well and is not
        supported — raising rather than silently computing the wrong thing.
        """
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

    def topological_order(self) -> list["Value"]:
        """Every value this one depends on, inputs before outputs.

        Backpropagation must visit a node only once **all** of its
        consumers have contributed their gradient. A topological sort,
        reversed, gives exactly that order. Doing it any other way is the
        classic source of gradients that are quietly too small.
        """
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
        """Fill in ``.grad`` on every value this one was computed from.

        Seeds ``self.grad = 1.0`` — the derivative of a value with respect
        to itself — then walks the graph in reverse topological order.

        Gradients **accumulate**, so calling ``backward()`` twice without
        ``zero_grad()`` doubles them. That is not a bug to fix here; it is
        the behaviour that makes gradient accumulation across mini-batches
        possible, and every framework works this way.
        """
        ordered = self.topological_order()
        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()

    def zero_grad(self) -> None:
        """Reset ``.grad`` to zero on this value and everything behind it."""
        for node in self.topological_order():
            node.grad = 0.0


def numerical_gradient(function, inputs: list[float], index: int, h: float = 1e-5) -> float:
    """Estimate ``d function / d inputs[index]`` by finite differences.

    Uses the **central** difference, ``(f(x+h) - f(x-h)) / 2h``, rather
    than the forward difference ``(f(x+h) - f(x)) / h``. The central form
    has error proportional to ``h²`` instead of ``h``, which buys several
    digits of agreement for free — and those digits are what make a
    gradient check meaningful rather than decorative.

    ``h`` around ``1e-5`` is the usual compromise: smaller and the
    subtraction loses precision to floating point, larger and the
    approximation itself drifts. Day 2 measures that trade-off directly.

    >>> round(numerical_gradient(lambda a, b: a * b, [2.0, 3.0], 0), 6)
    3.0
    """
    if not 0 <= index < len(inputs):
        raise IndexError(f"no input {index} among {len(inputs)}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    forward = list(inputs)
    backward = list(inputs)
    forward[index] += h
    backward[index] -= h
    return (function(*forward) - function(*backward)) / (2 * h)


def check_gradient(function, inputs: list[float], tolerance: float = 1e-6) -> list[float]:
    """Compare autograd's gradients against finite differences.

    Returns the analytic gradients and raises ``AssertionError`` on any
    that disagree. This is the repo's core habit in one function, and
    every later day calls something shaped like it.

    >>> check_gradient(lambda a, b: a * b + a, [2.0, 3.0])
    [4.0, 2.0]
    """
    values = [Value(x) for x in inputs]
    output = function(*values)
    if not isinstance(output, Value):
        raise TypeError("function must return a Value")
    output.backward()

    analytic = [v.grad for v in values]
    for i, gradient in enumerate(analytic):
        numeric = numerical_gradient(function, inputs, i)
        if abs(gradient - numeric) > tolerance * max(1.0, abs(numeric)):
            raise AssertionError(
                f"gradient {i} disagrees: autograd {gradient:.10f} "
                f"vs finite difference {numeric:.10f}"
            )
    return analytic


if __name__ == "__main__":
    print("a tiny graph, differentiated:")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    print(f"  a={a.data}  b={b.data}  c={c.data}")
    print(f"  d = a*b + c = {d.data}")
    d.backward()
    print(f"  dd/da = {a.grad}   (= b)")
    print(f"  dd/db = {b.grad}   (= a)")
    print(f"  dd/dc = {c.grad}   (= 1)")

    print("\nevery one of those checked against finite differences:")
    for name, function, inputs in (
        ("a*b + c", lambda x, y, z: x * y + z, [2.0, -3.0, 10.0]),
        ("a**3", lambda x: x ** 3, [2.0]),
        ("a/b", lambda x, y: x / y, [6.0, 3.0]),
        ("(a+b)*(a-b)", lambda x, y: (x + y) * (x - y), [5.0, 2.0]),
        ("a*a*a + 2*a", lambda x: x * x * x + 2 * x, [1.5]),
    ):
        gradients = check_gradient(function, inputs)
        print(f"  {name:<14} grad = {[round(g, 6) for g in gradients]}  ok")

    print("\na value used twice accumulates both contributions:")
    x = Value(3.0)
    y = x + x
    y.backward()
    print(f"  y = x + x;  dy/dx = {x.grad}  (2, not 1 - the += in _backward)")

    print("\nthe topological order, for x*x + x:")
    x = Value(2.0)
    z = x * x + x
    print(f"  {len(z.topological_order())} nodes, inputs first:")
    for node in z.topological_order():
        label = node._operation or "leaf"
        print(f"    {label:<6} data={node.data:g}")

    print("\nwhy the order matters:")
    print("  a node must not propagate its gradient until every consumer has")
    print("  contributed. Reverse topological order guarantees that; any other")
    print("  order silently produces gradients that are too small.")
