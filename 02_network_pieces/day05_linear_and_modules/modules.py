"""Day 5 — Linear layers, initialization, and a Module base.

Level 1 built an autograd. Everything from here is that autograd wearing
the shapes a practitioner actually handles.

Two ideas, and the second is bigger than it looks.

A **Module** is anything with parameters that need gradients. It has to
answer one question — *which tensors do you want trained?* — because the
optimizer of Day 7 needs that list and nothing else about the model. The
walk that answers it recurses through attributes and lists, and a
parameter it fails to find never trains, which surfaces as a model that
plateaus for no visible reason.

**Initialization** decides whether a deep network trains at all. The
weights are random, but the *scale* of that randomness is not free: it is
chosen to keep the variance of activations roughly constant with depth.
Too small and the signal vanishes through the stack; too large and tanh
saturates, which Day 2 already showed kills the gradient. This day
measures both failures over 30 layers rather than quoting the formulas.
"""

from __future__ import annotations

import math

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
        for node in ordered:
            node.grad = np.zeros_like(node.data)
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

class Module:
    """Base class: anything with parameters that need gradients.

    Two responsibilities, and no more. ``parameters()`` walks the object
    graph and returns every ``Tensor`` that should be trained, and
    ``zero_grad()`` clears them. The optimizer of Day 7 needs exactly this
    list and nothing else about the model.

    Sub-modules are found by inspecting ``__dict__``, so a layer stored as
    an attribute is discovered automatically and a layer stored in a bare
    list is **not** — hence ``ModuleList``. Silently missing a parameter
    means it never trains, which shows up as a model that plateaus for no
    visible reason.

    >>> layer = Linear(3, 2, seed=0)
    >>> [p.shape for p in layer.parameters()]
    [(3, 2), (2,)]
    """

    def parameters(self) -> list[Tensor]:
        found: list[Tensor] = []
        seen: set[int] = set()

        def collect(item) -> None:
            if isinstance(item, Tensor):
                if id(item) not in seen:
                    seen.add(id(item))
                    found.append(item)
            elif isinstance(item, Module):
                for value in vars(item).values():
                    collect(value)
            elif isinstance(item, (list, tuple)):
                for value in item:
                    collect(value)

        collect(self)
        return found

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = np.zeros_like(parameter.data)

    def parameter_count(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no forward()")


class ModuleList(Module):
    """A list of modules whose parameters are still discoverable.

    A plain Python list works — ``parameters()`` recurses into lists — but
    this makes the intent explicit and gives indexing and iteration.

    >>> stack = ModuleList([Linear(2, 2, seed=0), Linear(2, 2, seed=1)])
    >>> len(stack.parameters())
    4
    """

    def __init__(self, modules) -> None:
        self.modules = list(modules)

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, index):
        return self.modules[index]


def init_scale(fan_in: int, fan_out: int, mode: str = "xavier") -> float:
    """Standard deviation for a weight matrix of the given shape.

    The point of an initializer is to keep the **variance of activations**
    roughly constant as they pass through layers. Too small and the signal
    dies out over depth; too large and it explodes. Both are measured on
    this day rather than asserted.

    - ``xavier`` — ``sqrt(2 / (fan_in + fan_out))``. Balances the forward
      and backward passes; the right default for ``tanh``.
    - ``he`` — ``sqrt(2 / fan_in)``. ReLU zeroes half its inputs, so the
      variance needs twice the boost to compensate.
    - ``lecun`` — ``sqrt(1 / fan_in)``. Preserves forward variance only.

    >>> round(init_scale(100, 100, "he"), 4)
    0.1414
    """
    if fan_in < 1 or fan_out < 1:
        raise ValueError(f"fan_in and fan_out must be positive, got {fan_in}, {fan_out}")
    if mode == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if mode == "he":
        return math.sqrt(2.0 / fan_in)
    if mode == "lecun":
        return math.sqrt(1.0 / fan_in)
    raise ValueError(f"unknown init mode {mode!r}; expected xavier, he or lecun")


class Linear(Module):
    """``x @ W + b``, with a shaped initialization and an optional bias.

    ``W`` is ``(in_features, out_features)`` so a batch of rows stays a
    batch of rows — ``(batch, in) @ (in, out) -> (batch, out)`` — and no
    transpose appears anywhere in the forward pass.

    **The bias starts at zero.** It has no fan-in to balance and a random
    bias only adds noise to the first steps. The weights must *not* start
    at zero: identical weights receive identical gradients and stay
    identical for ever, so a layer of 100 units would learn exactly one
    thing. A test pins that symmetry failure.

    >>> layer = Linear(4, 3, seed=0)
    >>> layer(Tensor(np.zeros((2, 4)))).shape
    (2, 3)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "xavier", seed: int | None = None) -> None:
        if in_features < 1 or out_features < 1:
            raise ValueError(
                f"features must be positive, got {in_features} and {out_features}"
            )
        rng = np.random.default_rng(seed)
        scale = init_scale(in_features, out_features, init)
        self.weight = Tensor(rng.normal(0.0, scale, size=(in_features, out_features)))
        self.bias = Tensor(np.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape}"
            )
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Sequential(Module):
    """Apply modules in order.

    >>> net = Sequential(Linear(3, 4, seed=0), Tanh(), Linear(4, 2, seed=1))
    >>> net(Tensor(np.zeros((5, 3)))).shape
    (5, 2)
    """

    def __init__(self, *modules) -> None:
        self.layers = ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Tanh(Module):
    """Activation as a module, so it can sit inside ``Sequential``."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("a Module knows its own parameters:")
    net = Sequential(Linear(8, 16, seed=0), Tanh(), Linear(16, 4, seed=1))
    print(f"  {len(net.parameters())} tensors, {net.parameter_count():,} numbers")
    for parameter in net.parameters():
        print(f"    {str(parameter.shape):<12} {parameter.data.size:>5}")
    print("  the optimizer on Day 7 needs this list and nothing else about the model")

    print("\ninitialization decides whether a deep signal survives:")
    depth, width = 30, 128
    print(f"  {depth} tanh layers of width {width}, activation std at each depth:")
    print(f"  {'init':<10}{'std @1':>9}{'@10':>9}{'@20':>9}{'@30':>9}")
    for mode, factor in (("too small", 0.1), ("lecun", 1.0), ("xavier", 1.0), ("too large", 6.0)):
        x = Tensor(rng.normal(size=(64, width)))
        chosen = "lecun" if mode == "too small" or mode == "too large" else mode
        stds = []
        for layer_index in range(depth):
            scale = init_scale(width, width, chosen) * factor
            weight = Tensor(np.random.default_rng(layer_index).normal(0.0, scale, (width, width)))
            x = (x @ weight).tanh()
            if layer_index + 1 in (1, 10, 20, 30):
                stds.append(float(np.std(x.data)))
        print(f"  {mode:<10}" + "".join(f"{s:>9.4f}" for s in stds))
    print("  too small and the signal vanishes; too large and tanh saturates,")
    print("  which Day 2 already showed kills the gradient")

    print("\nzero weights are the one initialization that cannot work:")
    layer = Linear(4, 3, seed=0)
    layer.weight = Tensor(np.zeros((4, 3)))
    x = Tensor(rng.normal(size=(6, 4)))
    ((layer(x)) ** 2).sum().backward()
    columns = layer.weight.grad.T
    print(f"  every output unit receives the same gradient: "
          f"{np.allclose(columns[0], columns[1]) and np.allclose(columns[1], columns[2])}")
    print("  so they stay identical for ever, and a layer of 100 units learns one thing")

    print("\nbias starts at zero, weights do not:")
    layer = Linear(5, 3, seed=0)
    print(f"  bias   {layer.bias.data.round(3).tolist()}")
    print(f"  weight std {float(layer.weight.data.std()):.4f}  "
          f"(xavier target {init_scale(5, 3):.4f})")

    print("\nthe whole thing still differentiates, checked element by element:")
    for name, function, inputs in (
        ("Linear", lambda x, w, b: (x @ w + b).sum(),
         [rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=(2,))]),
        ("Linear+tanh", lambda x, w, b: (x @ w + b).tanh().sum(),
         [rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=(2,))]),
        ("two layers", lambda x, w1, w2: ((x @ w1).tanh() @ w2).sum(),
         [rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=(5, 2))]),
    ):
        check_gradient(function, inputs)
        print(f"  {name:<14} ok")

    print("\nparameters() finds a module in an attribute and in a ModuleList:")
    class Both(Module):
        def __init__(self):
            self.first = Linear(2, 2, seed=0)
            self.rest = ModuleList([Linear(2, 2, seed=1), Linear(2, 2, seed=2)])

    print(f"  {len(Both().parameters())} tensors found (3 layers x weight+bias = 6)")
    print("  a layer the walk misses never trains, and the model just plateaus")
