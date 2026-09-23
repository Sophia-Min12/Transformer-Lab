"""Day 3 — Tensor autograd: NumPy arrays and broadcasting.

A scalar autograd is correct and unusably slow. A single attention head
does millions of multiplications, and Python cannot run millions of
``Value`` objects per forward pass.

Replacing the scalar with a NumPy array changes remarkably little. The
graph, the topological sort and ``backward()`` are untouched — the chain
rule does not care about shapes. Only the ``_backward`` closures change,
and almost all of them only by adding one line.

That one line is the whole difficulty. **Broadcasting silently duplicates
data in the forward pass, and every duplicate must be summed back in the
backward pass.** Forget it and the gradient has the wrong shape, which
NumPy will often broadcast into something plausible rather than crashing —
so the model still trains, just wrongly. It is the tensor equivalent of
Day 1's ``+=`` bug, and it is caught the same way: by checking.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(gradient: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a gradient back down to ``shape``, undoing NumPy broadcasting.

    If the forward pass broadcast a ``(3,)`` against a ``(2, 3)``, that row
    was *used twice*, so its gradient is the sum of both contributions —
    exactly Day 1's rule that a value used twice accumulates, expressed in
    shapes.

    Two cases, in order:

    1. Broadcasting **prepends** axes, so any leading dimensions the
       original did not have are summed away entirely.
    2. Within the remaining axes, a dimension of size 1 was stretched, so
       it is summed with ``keepdims`` to stay size 1.

    >>> unbroadcast(np.ones((2, 3)), (3,))
    array([2., 2., 2.])
    >>> unbroadcast(np.ones((2, 3)), (1, 3))
    array([[2., 2., 2.]])
    """
    gradient = np.asarray(gradient, dtype=float)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """An array that records how it was computed.

    The same three fields as Day 1's ``Value`` — ``data``, ``grad``,
    ``_backward`` — with ``grad`` now an array of the same shape.

    >>> a = Tensor([1.0, 2.0])
    >>> b = Tensor([3.0, 4.0])
    >>> (a * b).data
    array([3., 8.])
    """

    def __init__(self, data, children: tuple = (), operation: str = "") -> None:
        if isinstance(data, Tensor):
            raise TypeError("Tensor takes an array, not another Tensor")
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._previous = set(children)
        self._operation = operation

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data})"

    @staticmethod
    def _wrap(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other) -> "Tensor":
        """Addition passes the gradient through — then unbroadcasts it."""
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + unbroadcast(out.grad, self.shape)
            other.grad = other.grad + unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        """Element-wise product; each factor is scaled by the other."""
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + unbroadcast(other.data * out.grad, self.shape)
            other.grad = other.grad + unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent) -> "Tensor":
        if isinstance(exponent, Tensor):
            raise TypeError("exponent must be a constant, not a Tensor")
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad = self.grad + exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    def __truediv__(self, other) -> "Tensor":
        return self * (self._wrap(other) ** -1.0)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other) -> "Tensor":
        return self._wrap(other) * (self ** -1.0)

    # ---- element-wise functions (reused from day02) --------------------

    def exp(self) -> "Tensor":
        value = np.exp(self.data)
        out = Tensor(value, (self,), "exp")

        def _backward():
            self.grad = self.grad + value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        if np.any(self.data <= 0):
            raise ValueError("log of a non-positive entry")
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad = self.grad + out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        value = np.tanh(self.data)
        out = Tensor(value, (self,), "tanh")

        def _backward():
            self.grad = self.grad + (1.0 - value * value) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out

    # ---- shape-changing operations -------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Sum, whose gradient is a broadcast back to the original shape.

        Summing and broadcasting are **adjoint**: the derivative of a sum
        is to copy the incoming gradient to every element that fed it, and
        the derivative of a broadcast is to sum. Seeing the pair once makes
        ``unbroadcast`` obvious rather than fiddly.

        >>> t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t.sum().data
        array(10.)
        """
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            gradient = out.grad
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis)
            self.grad = self.grad + np.broadcast_to(gradient, self.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Mean, which is a sum scaled by how many terms it averaged."""
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, *shape) -> "Tensor":
        """Reshape, whose gradient is the reverse reshape. No arithmetic."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad = self.grad + out.grad.reshape(original)

        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
        """Swap the last two axes; the gradient swaps them back."""
        axes = list(range(self.data.ndim))
        if len(axes) < 2:
            raise ValueError("transpose needs at least 2 dimensions")
        axes[-2], axes[-1] = axes[-1], axes[-2]
        out = Tensor(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            self.grad = self.grad + out.grad.transpose(axes)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ---- graph walk (reused from day01) --------------------------------

    def topological_order(self) -> list["Tensor"]:
        ordered: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: "Tensor") -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._previous:
                visit(child)
            ordered.append(node)

        visit(self)
        return ordered

    def backward(self) -> None:
        """Seed the output gradient with ones and walk the graph backwards.

        Requires a **scalar** output. A non-scalar has no single derivative
        to propagate — you would have to say which combination of its
        entries you meant, and silently summing them (which is what
        seeding with ones does) is a guess. Losses are scalars; call
        ``.sum()`` or ``.mean()`` first and mean it.
        """
        if self.data.size != 1:
            raise ValueError(
                f"backward() needs a scalar output, got shape {self.shape}; "
                "reduce with .sum() or .mean() first"
            )
        ordered = self.topological_order()
        for node in ordered:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(ordered):
            node._backward()

    def zero_grad(self) -> None:
        for node in self.topological_order():
            node.grad = np.zeros_like(node.data)


def numerical_gradient(function, inputs: list, h: float = 1e-5) -> list[np.ndarray]:
    """Finite-difference gradient of a scalar-valued function of arrays.

    Perturbs **one element at a time**, so the cost is two forward passes
    per parameter. That is precisely why nobody trains this way and why
    reverse-mode autodiff exists: backprop gets all of them in one pass.

    >>> gradients = numerical_gradient(lambda a: (a * a).sum(), [np.array([3.0])])
    >>> [round(g, 6) for g in gradients[0].tolist()]
    [6.0]
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    def evaluate(arrays) -> float:
        result = function(*[Tensor(a) for a in arrays])
        return float(np.asarray(result.data if isinstance(result, Tensor) else result))

    gradients = []
    arrays = [np.asarray(a, dtype=float) for a in inputs]
    for position, array in enumerate(arrays):
        gradient = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            forward = [a.copy() for a in arrays]
            backward = [a.copy() for a in arrays]
            forward[position][index] += h
            backward[position][index] -= h
            gradient[index] = (evaluate(forward) - evaluate(backward)) / (2 * h)
        gradients.append(gradient)
    return gradients


def check_gradient(function, inputs: list, tolerance: float = 1e-6, h: float = 1e-5):
    """Compare autograd against finite differences, element by element."""
    tensors = [Tensor(a) for a in inputs]
    output = function(*tensors)
    if not isinstance(output, Tensor):
        raise TypeError("function must return a Tensor")
    output.backward()

    analytic = [t.grad for t in tensors]
    numeric = numerical_gradient(function, inputs, h)
    for i, (mine, theirs) in enumerate(zip(analytic, numeric)):
        if mine.shape != theirs.shape:
            raise AssertionError(
                f"gradient {i} has shape {mine.shape}, expected {theirs.shape}"
            )
        scale = np.maximum(1.0, np.maximum(np.abs(mine), np.abs(theirs)))
        error = float(np.max(np.abs(mine - theirs) / scale))
        if error > tolerance:
            raise AssertionError(f"gradient {i} disagrees by {error:.2e}")
    return analytic


if __name__ == "__main__":
    print("the graph machinery is unchanged; only the closures know shapes.\n")

    print("broadcasting, forward and backward:")
    rows = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = Tensor([10.0, 20.0, 30.0])
    total = (rows + bias).sum()
    total.backward()
    print(f"  (2,3) + (3,) -> {(rows + bias).shape}")
    print(f"  d/d rows shape {rows.grad.shape}: every entry used once  -> {rows.grad[0]}")
    print(f"  d/d bias shape {bias.grad.shape}: each entry used twice  -> {bias.grad}")
    print("  the bias gradient is 2, not 1. Forgetting to sum is the classic bug,")
    print("  and NumPy would often broadcast the wrong-shaped result into silence.")

    print("\nsum and broadcast are adjoint:")
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t.sum().backward()
    print(f"  d(sum)/d t = {t.grad.tolist()}  (ones - the gradient is broadcast back)")

    print("\neverything checked element by element:")
    rng = np.random.default_rng(0)
    for name, function, inputs in (
        ("a + b (broadcast)", lambda a, b: (a + b).sum(), [rng.normal(size=(2, 3)), rng.normal(size=(3,))]),
        ("a * b (broadcast)", lambda a, b: (a * b).sum(), [rng.normal(size=(2, 3)), rng.normal(size=(1, 3))]),
        ("tanh", lambda a: a.tanh().sum(), [rng.normal(size=(3, 2))]),
        ("relu", lambda a: a.relu().sum(), [rng.normal(size=(4,))]),
        ("exp/log", lambda a: (a.exp() + 1.0).log().sum(), [rng.normal(size=(3,))]),
        ("mean", lambda a: a.mean(), [rng.normal(size=(2, 3))]),
        ("sum(axis=0)", lambda a: (a.sum(axis=0) ** 2).sum(), [rng.normal(size=(2, 3))]),
        ("reshape", lambda a: (a.reshape(3, 2) ** 2).sum(), [rng.normal(size=(2, 3))]),
        ("transpose", lambda a: (a.T * 2.0).sum(), [rng.normal(size=(2, 3))]),
        ("a used twice", lambda a: (a * a + a).sum(), [rng.normal(size=(3,))]),
    ):
        check_gradient(function, inputs)
        print(f"  {name:<20} ok")

    print("\nbackward() refuses a non-scalar output:")
    try:
        Tensor([1.0, 2.0]).backward()
    except ValueError as error:
        print(f"  ValueError: {error}")
    print("  seeding a vector with ones would silently mean 'sum of the entries',")
    print("  which is a guess about what you wanted. Losses are scalars; say so.")

    print("\nwhy reverse mode exists at all:")
    size = 200
    print(f"  a function of {size} parameters:")
    print(f"    finite differences: {2 * size} forward passes")
    print("    backprop:           1 forward + 1 backward")
