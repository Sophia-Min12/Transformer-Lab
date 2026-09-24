"""Day 4 — Matrix multiplication and the chain rule in two dimensions.

Day 3 could express a linear layer as ``(x * w).sum(axis=1)``. That is
correct, clumsy, and slow: it materialises an ``(n, in, out)`` array where
a matrix product needs none. Every real network is mostly ``@``.

Its derivative is the first in this repo worth deriving rather than
looking up, because the result is so short that it looks like a guess:

    C = A @ B        with A of shape (n, k) and B of shape (k, m)

    dL/dA = dL/dC @ Bᵀ
    dL/dB = Aᵀ @ dL/dC

Two matrix products, no sums, no loops. The derivation is in the README;
what matters here is that **the shapes force the answer**. ``dL/dA`` must
be ``(n, k)``, and the only way to build that from an ``(n, m)`` gradient
and a ``(k, m)`` matrix is to multiply by the transpose. Getting matmul
backward wrong is hard for exactly that reason — and the gradient check
catches the one case where it is easy.
"""

from __future__ import annotations

import numpy as np


# reused from day03
def unbroadcast(gradient: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a gradient back down to shape, undoing NumPy broadcasting."""
    gradient = np.asarray(gradient, dtype=float)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


# reused from day03
class Tensor:
    """An array that records how it was computed.

    >>> (Tensor([[1.0, 2.0]]) @ Tensor([[3.0], [4.0]])).data
    array([[11.]])
    """

    #: Tell NumPy not to try handling ``ndarray @ Tensor`` itself. Without
    #: this, ``np.ndarray.__matmul__`` attempts to coerce the Tensor and
    #: raises before Python ever falls back to ``__rmatmul__`` — so the
    #: reflected operators silently do not work with a raw array on the
    #: left. Setting it to None makes NumPy return NotImplemented, which
    #: is what triggers the fallback.
    __array_ufunc__ = None

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
        return f"Tensor(shape={self.shape})"

    @staticmethod
    def _wrap(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + unbroadcast(out.grad, self.shape)
            other.grad = other.grad + unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
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

    # ---- the day's operation -------------------------------------------

    def __matmul__(self, other) -> "Tensor":
        """``C = A @ B``, with ``dA = dC @ Bᵀ`` and ``dB = Aᵀ @ dC``.

        Only the **last two** axes take part in the product; anything in
        front is a batch dimension, which ``np.matmul`` broadcasts and
        ``unbroadcast`` then folds back. That is what makes multi-head
        attention (Day 11) a single call rather than a loop over heads.

        1-D operands are rejected. NumPy silently promotes them — a
        ``(k,)`` becomes ``(1, k)`` or ``(k, 1)`` depending on which side
        it is on, and then demotes the result — which quietly changes what
        the gradient shapes mean. Asking the caller to be explicit costs a
        ``reshape`` and removes a whole class of confusion.

        >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = Tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> (a @ b).data
        array([[1., 2.],
               [3., 4.]])
        """
        other = self._wrap(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError(
                f"matmul needs 2-D operands or better, got {self.shape} @ {other.shape}; "
                "reshape a vector to (1, k) or (k, 1) and say which you meant"
            )
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(
                f"shapes {self.shape} and {other.shape} do not line up: "
                f"{self.shape[-1]} != {other.shape[-2]}"
            )
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
            grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad = self.grad + unbroadcast(grad_self, self.shape)
            other.grad = other.grad + unbroadcast(grad_other, other.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other) -> "Tensor":
        return self._wrap(other) @ self

    # ---- element-wise (reused from day02/day03) ------------------------

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

    # ---- shape (reused from day03) -------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            gradient = out.grad
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis)
            self.grad = self.grad + np.broadcast_to(gradient, self.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad = self.grad + out.grad.reshape(original)

        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
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
        if self.data.size != 1:
            raise ValueError(
                f"backward() needs a scalar output, got shape {self.shape}; "
                "reduce with .sum() or .mean() first"
            )
        ordered = self.topological_order()
        self.grad = np.ones_like(self.data)
        for node in reversed(ordered):
            node._backward()

    def zero_grad(self) -> None:
        for node in self.topological_order():
            node.grad = np.zeros_like(node.data)


# reused from day03
def numerical_gradient(function, inputs: list, h: float = 1e-5) -> list[np.ndarray]:
    """Finite-difference gradient of a scalar-valued function of arrays."""
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


# reused from day03
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


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``x @ W + b`` — the layer every network is mostly made of.

    Three days of machinery reduce to one line, and its gradient comes out
    of ``@`` and ``+`` without anything new being written.
    """
    return x @ weight + bias


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("the shapes force the answer:")
    a = Tensor(rng.normal(size=(4, 3)))
    b = Tensor(rng.normal(size=(3, 5)))
    c = a @ b
    c.sum().backward()
    print(f"  A {a.shape} @ B {b.shape} = C {c.shape}")
    print(f"  dL/dA must be {a.shape} = dC {c.shape} @ Bt {(b.shape[1], b.shape[0])}")
    print(f"  dL/dB must be {b.shape} = At {(a.shape[1], a.shape[0])} @ dC {c.shape}")
    print(f"  and it is: {a.grad.shape} and {b.grad.shape}")

    print("\na linear layer, and the gradients of all three inputs:")
    x = Tensor(rng.normal(size=(8, 4)))
    weight = Tensor(rng.normal(size=(4, 3)) * 0.1)
    bias = Tensor(np.zeros(3))
    loss = (linear(x, weight, bias).tanh() ** 2).sum()
    loss.backward()
    print(f"  x {x.shape} @ W {weight.shape} + b {bias.shape}")
    print(f"  grads: x {x.grad.shape}, W {weight.grad.shape}, b {bias.grad.shape}")
    print(f"  the bias gradient summed over all {x.shape[0]} rows, as broadcasting requires")

    print("\nevery one checked against finite differences:")
    for name, function, inputs in (
        ("A @ B", lambda p, q: (p @ q).sum(), [rng.normal(size=(3, 4)), rng.normal(size=(4, 2))]),
        ("linear layer", lambda p, q, r: linear(p, q, r).sum(),
         [rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), rng.normal(size=(2,))]),
        ("two layers", lambda p, q, r: ((p @ q).tanh() @ r).sum(),
         [rng.normal(size=(4, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 2))]),
        ("A @ At", lambda p: (p @ p.T).sum(), [rng.normal(size=(3, 4))]),
        ("batched", lambda p, q: (p @ q).sum(), [rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 4, 3))]),
        ("batch broadcast", lambda p, q: (p @ q).sum(),
         [rng.normal(size=(2, 3, 4)), rng.normal(size=(4, 3))]),
        ("squared error", lambda p, q, t: (((p @ q) - t) ** 2).mean(),
         [rng.normal(size=(6, 3)), rng.normal(size=(3, 2)), rng.normal(size=(6, 2))]),
    ):
        check_gradient(function, inputs)
        print(f"  {name:<18} ok")

    print("\nA @ A.T uses A twice, and the gradient must collect both paths:")
    m = Tensor(rng.normal(size=(3, 4)))
    (m @ m.T).sum().backward()
    numeric = numerical_gradient(lambda p: (p @ p.T).sum(), [m.data])[0]
    print(f"  max disagreement with finite differences: {np.abs(m.grad - numeric).max():.2e}")

    print("\n1-D operands are rejected rather than silently promoted:")
    try:
        Tensor([1.0, 2.0]) @ Tensor([[1.0], [2.0]])
    except ValueError as error:
        print(f"  ValueError: {error}")

    print("\nmismatched inner dimensions are named, not left to NumPy:")
    try:
        Tensor(np.ones((3, 4))) @ Tensor(np.ones((5, 2)))
    except ValueError as error:
        print(f"  ValueError: {error}")

    print("\nspeed, against day 3's stand-in:")
    import time
    big_x = np.random.default_rng(1).normal(size=(256, 256))
    big_w = np.random.default_rng(2).normal(size=(256, 256))
    start = time.perf_counter()
    (Tensor(big_x) @ Tensor(big_w)).sum().backward()
    fast = time.perf_counter() - start
    start = time.perf_counter()
    (Tensor(big_x).reshape(256, 256, 1) * Tensor(big_w).reshape(1, 256, 256)).sum().backward()
    slow = time.perf_counter() - start
    print(f"  x @ w                       {fast * 1000:7.1f} ms")
    print(f"  (x * w).sum() over an axis  {slow * 1000:7.1f} ms   ({slow / fast:.0f}x slower)")
    print("  and the second one allocates a 256x256x256 array to do it")
