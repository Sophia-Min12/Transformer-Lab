"""Day 17 - PyTorch comparison, profiling, and the honest writeup.

Sixteen days of hand-derived calculus, checked for the first time against
an implementation written by someone else. Then a profile of where the
time goes, and a plain account of what this repo does and does not show.

PyTorch is the only third-party dependency in the repo and it is needed
only here; every section degrades to a clear message without it.

Run the demo:      python comparison.py
Run the tests:     python -m unittest        (from this folder)
"""

from __future__ import annotations

import cProfile
import math
import pathlib
import pstats
import sys
import time

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

def naive_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """``exp(x) / sum(exp(x))`` — the definition, and unusable.

    Kept in the file on purpose so the overflow can be demonstrated
    rather than described. ``exp(1000)`` is ``inf``, ``inf/inf`` is
    ``nan``, and a single ``nan`` reaches every parameter in one backward
    pass.
    """
    scaled = np.exp(np.asarray(logits, dtype=float))
    return scaled / scaled.sum(axis=axis, keepdims=True)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Stable softmax: subtract the row maximum before exponentiating.

    ``softmax(x) == softmax(x - c)`` for **any** constant ``c``, because
    the shift cancels between numerator and denominator. Choosing
    ``c = max(x)`` makes the largest exponent exactly ``exp(0) = 1``, so
    nothing can overflow and the smallest terms underflow harmlessly to 0.

    The shift is deliberately **not** part of the graph — it is computed
    from ``x.data`` and wrapped as a constant. That is not a shortcut: the
    output genuinely does not depend on ``c``, so its derivative with
    respect to ``c`` is zero, and routing a gradient through the max would
    add a term that must cancel to nothing anyway.

    >>> softmax(Tensor([1.0, 1.0, 1.0])).data.round(6).tolist()
    [0.333333, 0.333333, 0.333333]
    """
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    scaled = (x - shift).exp()
    return scaled / scaled.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """``log(softmax(x))``, computed without ever forming ``softmax(x)``.

    Taking the log of a softmax output is the mistake this function
    exists to prevent. A confident model produces probabilities like
    ``1e-30``; in float64 a small enough one rounds to exactly ``0`` and
    ``log(0)`` is ``-inf``. The loss becomes infinite, the gradient
    becomes ``nan``, and the run is over.

    Computing ``x - c - log(sum(exp(x - c)))`` never forms the small
    probability at all — the subtraction happens in log space where
    ``-70`` is an ordinary number.

    >>> log_softmax(Tensor([0.0, 0.0])).data.round(6).tolist()
    [-0.693147, -0.693147]
    """
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    shifted = x - shift
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def one_hot(targets, classes: int) -> np.ndarray:
    """Integer class indices to a ``(n, classes)`` indicator matrix.

    >>> one_hot([0, 2], 3).tolist()
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    """
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1-D class indices, got shape {targets.shape}")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError(f"targets must be integers, got {targets.dtype}")
    if targets.size and (targets.min() < 0 or targets.max() >= classes):
        raise IndexError(f"target outside 0..{classes - 1}")
    encoded = np.zeros((targets.size, classes), dtype=float)
    encoded[np.arange(targets.size), targets] = 1.0
    return encoded


def cross_entropy(logits: Tensor, targets, reduction: str = "mean") -> Tensor:
    """Mean negative log-likelihood of the correct class.

    Takes **logits**, not probabilities. Every serious implementation does,
    for the reason in ``log_softmax``: handing this function a softmax
    output would mean it had to take a log of something already collapsed
    to zero. Fusing the two is what keeps the whole thing finite.

    >>> round(float(cross_entropy(Tensor([[0.0, 0.0]]), [0]).data), 6)
    0.693147
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"unknown reduction {reduction!r}; expected mean or sum")
    if logits.data.ndim != 2:
        raise ValueError(f"expected (batch, classes) logits, got shape {logits.shape}")

    indicator = Tensor(one_hot(targets, logits.shape[1]))
    picked = (indicator * log_softmax(logits, axis=-1)).sum()
    total = -picked
    return total * (1.0 / logits.shape[0]) if reduction == "mean" else total


def cross_entropy_gradient(logits: np.ndarray, targets) -> np.ndarray:
    """The analytic gradient of mean cross-entropy: ``(p - y) / n``.

    Worth writing out because it is startlingly simple. The softmax
    derivative is a full Jacobian and the log derivative is a reciprocal,
    and composing them cancels almost everything — what survives is
    "predicted probability minus the truth", scaled by the batch size.

    This is why the two are always fused in practice: the fused backward
    pass is one subtraction, while the honest chain through both would
    build and multiply a ``(classes, classes)`` matrix per example.

    A test checks this against what the autograd computes.
    """
    probabilities = naive_softmax(np.asarray(logits, dtype=float) -
                                  np.asarray(logits, dtype=float).max(axis=-1, keepdims=True))
    return (probabilities - one_hot(targets, probabilities.shape[1])) / len(probabilities)


def accuracy(logits: Tensor, targets) -> float:
    """Share of rows whose highest logit is the correct class."""
    predicted = np.argmax(logits.data, axis=-1)
    return float(np.mean(predicted == np.asarray(targets)))

class Optimizer:
    """Base class: holds the parameter list and clears its gradients.

    The optimizer never sees the model. It receives a list of tensors
    from ``Module.parameters()`` and updates them in place — which is why
    Day 5 spent its effort on making that list complete and correct.
    """

    def __init__(self, parameters, lr: float) -> None:
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimizer got no parameters; check Module.parameters()")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)
        self.steps = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

    Plain SGD is ``p -= lr * g``. Momentum accumulates a running velocity
    ``v = mu*v + g`` and steps along that instead, which damps the
    zig-zagging that happens when one direction of the loss surface is far
    steeper than another — the same ill-conditioning ArgMin-Lab studies
    directly.

    >>> p = Tensor([1.0]); p.grad = np.array([2.0])
    >>> optimizer = SGD([p], lr=0.1); optimizer.step()
    >>> p.data.round(6).tolist()
    [0.8]
    """

    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = float(momentum)
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.steps += 1
        for index, parameter in enumerate(self.parameters):
            if self.momentum:
                self.velocity[index] = self.momentum * self.velocity[index] + parameter.grad
                update = self.velocity[index]
            else:
                update = parameter.grad
            parameter.data = parameter.data - self.lr * update


class Adam(Optimizer):
    """Adam, with the bias correction spelled out.

    Two running averages per parameter: ``m`` of the gradient and ``v`` of
    its square. The update divides one by the square root of the other, so
    each parameter gets a step scaled to its own recent gradient
    magnitude — which is why Adam barely notices badly scaled features
    while SGD struggles with them.

    **Both averages start at zero**, which biases them toward zero for the
    first several steps. The correction divides by ``1 - beta**t``. It is
    not cosmetic and it does not do what people usually assume: at ``t=1``
    the *uncorrected* ratio is ``(1-b1)/sqrt(1-b2) ≈ 3.16``, so the first
    step is **three times too large**, not too small. The demo measures it.

    >>> p = Tensor([1.0]); p.grad = np.array([2.0])
    >>> optimizer = Adam([p], lr=0.1); optimizer.step()
    >>> p.data.round(6).tolist()
    [0.9]
    """

    def __init__(self, parameters, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 bias_correction: bool = True) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {beta1} and {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.bias_correction = bool(bias_correction)
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.steps += 1
        t = self.steps
        for index, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * gradient
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * gradient ** 2
            if self.bias_correction:
                m_hat = self.m[index] / (1 - self.beta1 ** t)
                v_hat = self.v[index] / (1 - self.beta2 ** t)
            else:
                m_hat, v_hat = self.m[index], self.v[index]
            parameter.data = parameter.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(model: Module, x: Tensor, targets, optimizer: Optimizer,
          steps: int = 200, loss_fn=None) -> list[float]:
    """Full-batch training loop. Returns the loss at each step.

    Four lines, in the order that matters: forward, zero, backward, step.

    **Zeroing before backward, not after**, is the convention that cannot
    go wrong. Gradients accumulate by design (Day 1), so a loop that
    forgets to clear them sums every step's gradient into the next — the
    model still trains, just with an ever-growing effective learning rate,
    and nothing reports it.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    loss_fn = loss_fn or cross_entropy
    history = []
    for _ in range(steps):
        loss = loss_fn(model(x), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.data))
    return history


def xor_dataset(n: int = 200, noise: float = 0.15, seed: int = 0):
    """The smallest problem a linear model provably cannot solve.

    Four clusters at the corners of a square, labelled by the XOR of their
    signs. No straight line separates them, so a model that reaches high
    accuracy here has genuinely learned a nonlinear boundary rather than
    fitting a slope.
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0])
    index = rng.integers(0, 4, size=n)
    points = corners[index] + rng.normal(0.0, noise, size=(n, 2))
    return Tensor(points), labels[index]

def take_rows(matrix: Tensor, indices) -> Tensor:
    """Select rows of ``matrix`` by integer index. The embedding lookup.

    Forward is ``matrix.data[indices]`` — a pure memory read, no
    arithmetic. The backward pass is the interesting half, and it contains
    the one mistake that is easy to make here.

    A gradient must be **scattered back** to the rows that were read, and
    a row read twice must receive both contributions. That is Day 1's
    ``+=`` rule again, in index form:

        grad[indices] += out.grad     # WRONG: repeated indices overwrite
        np.add.at(grad, indices, out.grad)   # right: they accumulate

    NumPy's fancy-index assignment keeps only the *last* write when an
    index repeats. In a language model the common words repeat constantly
    inside a single batch, so the wrong form silently under-counts exactly
    the rows that matter most — and the model still trains.

    >>> table = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> take_rows(table, [2, 0]).data.tolist()
    [[5.0, 6.0], [1.0, 2.0]]
    """
    indices = np.asarray(indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"indices must be integers, got {indices.dtype}")
    if matrix.data.ndim != 2:
        raise ValueError(f"expected a 2-D table, got shape {matrix.shape}")
    if indices.size and (indices.min() < 0 or indices.max() >= matrix.shape[0]):
        raise IndexError(f"index outside 0..{matrix.shape[0] - 1}")

    out = Tensor(matrix.data[indices], (matrix,), "take_rows")

    def _backward():
        scattered = np.zeros_like(matrix.data)
        np.add.at(scattered, indices, out.grad)
        matrix.grad = matrix.grad + scattered

    out._backward = _backward
    return out


class Embedding(Module):
    """A lookup table: one learned vector per symbol.

    Mathematically identical to multiplying a one-hot matrix by a weight
    matrix, and the reason nobody does it that way is arithmetic. For a
    vocabulary of 50,000 and a batch of 1,024 tokens, the one-hot form
    multiplies a ``(1024, 50000)`` matrix by a ``(50000, 768)`` one — 39
    billion multiply-adds, of which all but 786,432 are by zero. The
    lookup reads 1,024 rows.

    The demo measures both.

    >>> table = Embedding(10, 4, seed=0)
    >>> table([1, 5, 1]).shape
    (3, 4)
    """

    def __init__(self, vocab_size: int, dim: int, seed: int | None = None) -> None:
        if vocab_size < 1 or dim < 1:
            raise ValueError(f"vocab_size and dim must be positive, got {vocab_size}, {dim}")
        rng = np.random.default_rng(seed)
        # Embeddings are not a fan-in/fan-out layer - each row is read
        # independently - so a small fixed scale is the usual choice
        # rather than xavier.
        self.weight = Tensor(rng.normal(0.0, 0.02, size=(vocab_size, dim)))
        self.vocab_size = vocab_size
        self.dim = dim

    def forward(self, indices) -> Tensor:
        indices = np.asarray(indices)
        flat = take_rows(self.weight, indices.reshape(-1))
        return flat.reshape(*indices.shape, self.dim)


def one_hot_matmul(table: Tensor, indices, vocab_size: int) -> Tensor:
    """The equivalent one-hot formulation, kept for comparison only.

    Exists so the demo and the tests can show it produces **identical**
    values and identical gradients, at a cost that scales with the whole
    vocabulary instead of with the batch.
    """
    indices = np.asarray(indices).reshape(-1)
    encoded = np.zeros((indices.size, vocab_size))
    encoded[np.arange(indices.size), indices] = 1.0
    return Tensor(encoded) @ table

def attention_scores(query: Tensor, key: Tensor, scale: bool = True) -> Tensor:
    """``Q Kᵀ / √d_k`` — how much each position wants each other position.

    The dot product of two independent vectors with unit-variance entries
    has variance ``d_k``, so its **standard deviation grows as √d_k**. At
    ``d_k = 64`` the raw scores are spread over roughly ±8, and at 512
    over ±23.

    Softmax of numbers that far apart is a one-hot vector. That is not a
    numerical problem — Day 6's shift keeps it finite — it is a *learning*
    problem: the gradient of a saturated softmax is almost exactly zero,
    so the layer stops receiving any signal at all. Dividing by ``√d_k``
    returns the scores to unit variance whatever the dimension.

    The demo measures all three consequences: the spread, the collapse of
    the attention distribution, and the gradient that disappears with it.

    >>> q = Tensor([[1.0, 0.0]])
    >>> k = Tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> attention_scores(q, k).data.round(6).tolist()
    [[0.707107, 0.0]]
    """
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query and key must share their last dimension, "
            f"got {query.shape} and {key.shape}"
        )
    scores = query @ key.T
    if scale:
        scores = scores * (1.0 / math.sqrt(query.shape[-1]))
    return scores


def attention(query: Tensor, key: Tensor, value: Tensor,
              mask: np.ndarray | None = None, scale: bool = True):
    """Scaled dot-product attention. Returns ``(output, weights)``.

    ``softmax(Q Kᵀ / √d_k + mask) V``

    The weights are returned alongside the output because they are the
    only interpretable thing in a transformer, and Day 10 needs to inspect
    them to show the mask working.

    What makes this different from every layer so far: the coefficients
    applied to ``V`` are **computed from the input**, not learned and
    fixed. A ``Linear`` applies the same weights to every example; this
    computes a fresh set of weights for every position of every example.

    >>> q = k = v = Tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> out, weights = attention(q, k, v)
    >>> weights.data.sum(axis=-1).round(6).tolist()
    [1.0, 1.0]
    """
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key and value must have the same number of positions, "
            f"got {key.shape} and {value.shape}"
        )
    scores = attention_scores(query, key, scale)
    if mask is not None:
        mask = np.asarray(mask, dtype=float)
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError(
                f"mask shape {mask.shape} does not fit scores {scores.shape}"
            )
        scores = scores + Tensor(mask)
    weights = softmax(scores, axis=-1)
    return weights @ value, weights


def attention_entropy(weights: Tensor) -> np.ndarray:
    """Entropy of each attention row, in nats.

    A useful one-number summary of how spread out the attention is.
    ``log(T)`` means uniform over all ``T`` positions; ``0`` means the row
    has collapsed onto a single position and the layer is doing a lookup
    rather than a weighted average.
    """
    probabilities = np.clip(np.asarray(weights.data, dtype=float), 1e-300, None)
    return -np.sum(probabilities * np.log(probabilities), axis=-1)


class SelfAttention(Module):
    """Single-head self-attention: one input, three projections.

    ``Q``, ``K`` and ``V`` are all linear functions of the *same* input,
    which is what makes it *self*-attention. The three projections are
    where the learning happens — the attention mechanism itself has no
    parameters at all.

    >>> layer = SelfAttention(dim=8, seed=0)
    >>> out, weights = layer(Tensor(np.zeros((5, 8))))
    >>> out.shape, weights.shape
    ((5, 8), (5, 5))
    """

    def __init__(self, dim: int, head_dim: int | None = None, seed: int | None = None) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.head_dim = head_dim or dim
        self.query = Linear(dim, self.head_dim, bias=False, seed=seed)
        self.key = Linear(dim, self.head_dim, bias=False,
                          seed=None if seed is None else seed + 1)
        self.value = Linear(dim, self.head_dim, bias=False,
                            seed=None if seed is None else seed + 2)

    def forward(self, x: Tensor, mask: np.ndarray | None = None):
        return attention(self.query(x), self.key(x), self.value(x), mask=mask)

#: Large enough to make ``exp`` underflow to zero, small enough that
#: adding it to a finite score never produces ``inf - inf``. See
#: ``causal_mask`` for why ``-inf`` itself is the wrong choice.
MASK_VALUE = -1e9


def causal_mask(length: int, value: float = MASK_VALUE) -> np.ndarray:
    """An additive mask forbidding every position from seeing later ones.

    Zero on and below the diagonal, ``value`` above it. Added to the
    scores **before** the softmax, so the forbidden entries exponentiate
    to ``exp(-1e9) = 0`` and the surviving weights renormalize to sum to
    one on their own. Nothing downstream needs to know a mask was applied.

    **Why not ``-inf``?** It works here, and it stops working the moment a
    row is entirely masked — which happens in practice with padding masks,
    where a padded query attends to nothing. Then every score in the row
    is ``-inf``, Day 6's shift subtracts ``-inf``, and ``-inf - -inf`` is
    ``nan``. A large finite number degrades to a uniform row instead of
    poisoning the batch. A test pins both behaviours.

    >>> causal_mask(3, value=-1.0).tolist()
    [[0.0, -1.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]
    """
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    return np.triu(np.full((length, length), float(value)), k=1)


def causal_attention(query: Tensor, key: Tensor, value: Tensor):
    """Attention that may only look backwards. Returns ``(output, weights)``."""
    if query.shape[-2] != key.shape[-2]:
        raise ValueError(
            "causal attention needs as many queries as keys, "
            f"got {query.shape} and {key.shape}"
        )
    return attention(query, key, value, mask=causal_mask(query.shape[-2]))


def depends_on(output: Tensor, inputs: Tensor) -> np.ndarray:
    """Which input positions each output position actually depends on.

    Returns a boolean ``(queries, keys)`` matrix. Computed from the
    **gradient**, not from the attention weights: a weight of exactly zero
    is suggestive, but a non-zero gradient is proof that changing an input
    changes an output. This is the only honest way to test causality,
    because it cannot be fooled by a mask that was applied in the wrong
    place or to the wrong axis.
    """
    rows = output.shape[-2]
    influence = np.zeros((rows, inputs.shape[-2]), dtype=bool)
    for row in range(rows):
        inputs.grad = np.zeros_like(inputs.data)
        selector = np.zeros(output.shape)
        selector[row] = 1.0
        (output * Tensor(selector)).sum().backward()
        influence[row] = np.abs(inputs.grad).sum(axis=-1) > 1e-12
    return influence


class CausalSelfAttention(Module):
    """Self-attention restricted to the past and the present.

    >>> layer = CausalSelfAttention(dim=4, seed=0)
    >>> out, weights = layer(Tensor(np.zeros((3, 4))))
    >>> weights.data.round(3).tolist()
    [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.333, 0.333, 0.333]]
    """

    def __init__(self, dim: int, head_dim: int | None = None, seed: int | None = None) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.head_dim = head_dim or dim
        self.query = Linear(dim, self.head_dim, bias=False, seed=seed)
        self.key = Linear(dim, self.head_dim, bias=False,
                          seed=None if seed is None else seed + 1)
        self.value = Linear(dim, self.head_dim, bias=False,
                            seed=None if seed is None else seed + 2)

    def forward(self, x: Tensor):
        return causal_attention(self.query(x), self.key(x), self.value(x))

def permute(x: Tensor, axes) -> Tensor:
    """Reorder axes. The gradient permutes back by the inverse order.

    Needed to turn ``(time, heads, head_dim)`` into
    ``(heads, time, head_dim)`` so that Day 4's batched matmul treats the
    head axis as a batch and runs every head at once.

    Like ``reshape`` and ``transpose``, the backward pass contains **no
    arithmetic** — a permutation moves numbers, so its derivative moves
    them back.

    >>> permute(Tensor(np.zeros((2, 3, 4))), (1, 0, 2)).shape
    (3, 2, 4)
    """
    axes = tuple(axes)
    if sorted(axes) != list(range(x.data.ndim)):
        raise ValueError(f"axes {axes} is not a permutation of {x.data.ndim} dimensions")
    inverse = np.argsort(axes)
    out = Tensor(x.data.transpose(axes), (x,), "permute")

    def _backward():
        x.grad = x.grad + out.grad.transpose(inverse)

    out._backward = _backward
    return out


def split_heads(x: Tensor, heads: int) -> Tensor:
    """``(time, dim)`` -> ``(heads, time, head_dim)``.

    The dimension is **sliced**, not copied. Each head gets ``dim/heads``
    of the coordinates, so the total work is identical to a single head of
    the full width — multi-head attention is free.

    >>> split_heads(Tensor(np.zeros((5, 8))), heads=2).shape
    (2, 5, 4)
    """
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    time, dim = x.shape[-2], x.shape[-1]
    if dim % heads:
        raise ValueError(f"dim {dim} is not divisible by heads {heads}")
    return permute(x.reshape(time, heads, dim // heads), (1, 0, 2))


def merge_heads(x: Tensor) -> Tensor:
    """``(heads, time, head_dim)`` -> ``(time, dim)``. The inverse of split.

    >>> merge_heads(Tensor(np.zeros((2, 5, 4)))).shape
    (5, 8)
    """
    heads, time, head_dim = x.shape
    return permute(x, (1, 0, 2)).reshape(time, heads * head_dim)


class MultiHeadAttention(Module):
    """Several attentions in parallel, on slices of the same vectors.

    One head computes **one** distribution over positions, and therefore
    expresses one relationship: "the adjective before me" *or* "the
    subject of this clause", never both at once. The output is a single
    mixing matrix applied to every output coordinate alike.

    Splitting the width into ``h`` heads gives ``h`` independent
    distributions, each governing its own slice of the output. The demo
    builds a task that requires exactly this and shows one head failing at
    it — not slowly, but structurally.

    The cost is nothing. ``h`` heads of width ``d/h`` do the same number
    of multiplications as one head of width ``d``, and Day 4's matmul
    already batches over leading axes, so they run in one call.

    >>> layer = MultiHeadAttention(dim=8, heads=2, seed=0)
    >>> out, weights = layer(Tensor(np.zeros((5, 8))))
    >>> out.shape, weights.shape
    ((5, 8), (2, 5, 5))
    """

    def __init__(self, dim: int, heads: int = 1, causal: bool = False,
                 seed: int | None = None) -> None:
        if dim < 1 or heads < 1:
            raise ValueError(f"dim and heads must be positive, got {dim}, {heads}")
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.causal = causal
        self.query = Linear(dim, dim, bias=False, seed=seed)
        self.key = Linear(dim, dim, bias=False, seed=None if seed is None else seed + 1)
        self.value = Linear(dim, dim, bias=False, seed=None if seed is None else seed + 2)
        # The output projection is what lets the heads talk to each other.
        # Without it every head's slice would go straight to the next layer
        # untouched by the others, and the heads could never combine.
        self.out = Linear(dim, dim, bias=False, seed=None if seed is None else seed + 3)

    def forward(self, x: Tensor):
        time = x.shape[-2]
        q = split_heads(self.query(x), self.heads)
        k = split_heads(self.key(x), self.heads)
        v = split_heads(self.value(x), self.heads)
        mask = causal_mask(time) if self.causal else None
        head_out, weights = attention(q, k, v, mask=mask)
        return self.out(merge_heads(head_out)), weights


def two_relation_task(sequences: int = 260, length: int = 8, dim: int = 8, seed: int = 0):
    """A task needing two heads. Returns a list of ``(x, target)`` pairs.

    Two *content-addressed* lookups, both constant across positions:

        target[:, 0] = x[argmax(x[:, 2]), 0]
        target[:, 1] = x[argmax(x[:, 3]), 1]

    Feature 2 flags one position and feature 3 flags another. A single
    head produces one weight vector per query and applies it to every
    output coordinate alike, so it cannot read coordinate 0 from the first
    flagged position while reading coordinate 1 from the second.

    Two earlier versions of this task were wrong, and both failures are
    described in the README:

    - **One fixed sequence.** A single head reached zero loss by
      memorising it outright — attention weights are computed from ``x``,
      so a single example leaves ample room to encode the answer.
    - **A positional rule** (``target[t, 0] = x[t-1, 0]``). Attention is
      permutation-equivariant and has no idea what position it is at until
      Day 12, so nothing could solve it. Making the lookups content-based
      removes that confound.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(sequences):
        x = rng.normal(size=(length, dim))
        first = int(np.argmax(x[:, 2]))
        second = int(np.argmax(x[:, 3]))
        target = np.zeros((length, 2))
        target[:, 0] = x[first, 0]
        target[:, 1] = x[second, 1]
        pairs.append((Tensor(x), Tensor(target)))
    return pairs

def sinusoidal_encoding(length: int, dim: int) -> np.ndarray:
    """The fixed encoding from *Attention Is All You Need*.

    Position ``pos``, dimension ``i``:

        even i:  sin(pos / 10000^(i/d))
        odd  i:  cos(pos / 10000^((i-1)/d))

    Each pair of dimensions is a rotating clock hand, and the hands run at
    geometrically spaced rates — the fastest flips every couple of
    positions, the slowest barely moves across the whole sequence. Reading
    all of them at once identifies a position the way reading every hand
    of a clock identifies a time.

    Two properties come free and are worth the choice:

    - **No parameters, and no maximum length.** Position 5,000 has an
      encoding even if training never went past 512. A learned table
      simply has no row for it.
    - **Relative position is a linear function of absolute position.**
      ``PE(pos + k)`` is a fixed rotation of ``PE(pos)``, the same rotation
      for every ``pos``, so a layer can learn "three positions back" once
      rather than separately at every location.

    >>> sinusoidal_encoding(1, 4).round(4).tolist()
    [[0.0, 1.0, 0.0, 1.0]]
    """
    if length < 1 or dim < 1:
        raise ValueError(f"length and dim must be positive, got {length}, {dim}")
    if dim % 2:
        raise ValueError(f"dim must be even so sin/cos pair up, got {dim}")

    positions = np.arange(length, dtype=float)[:, None]
    pair_index = np.arange(0, dim, 2, dtype=float)[None, :]
    rates = positions / np.power(10000.0, pair_index / dim)
    encoding = np.zeros((length, dim))
    encoding[:, 0::2] = np.sin(rates)
    encoding[:, 1::2] = np.cos(rates)
    return encoding


class SinusoidalPositionalEncoding(Module):
    """Adds the fixed encoding to its input. No parameters.

    >>> layer = SinusoidalPositionalEncoding(dim=4)
    >>> layer(Tensor(np.zeros((3, 4)))).shape
    (3, 4)
    >>> layer.parameters()
    []
    """

    def __init__(self, dim: int, max_length: int = 4096) -> None:
        self.dim = dim
        self.table = sinusoidal_encoding(max_length, dim)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-2]
        if length > len(self.table):
            # Extend rather than fail: the formula is defined everywhere,
            # which is the whole argument for it over a learned table.
            self.table = sinusoidal_encoding(length, self.dim)
        return x + Tensor(self.table[:length])


class LearnedPositionalEncoding(Module):
    """One trained vector per position — what GPT-2 actually uses.

    Strictly more expressive than the sinusoids and strictly less general:
    the table has ``max_length`` rows and **nothing beyond**. A model
    trained at 512 positions cannot be asked about position 512, and the
    failure is an ``IndexError`` rather than a wrong answer, which is the
    better of the two.

    >>> layer = LearnedPositionalEncoding(dim=4, max_length=8, seed=0)
    >>> layer(Tensor(np.zeros((3, 4)))).shape
    (3, 4)
    """

    def __init__(self, dim: int, max_length: int = 512, seed: int | None = None) -> None:
        if dim < 1 or max_length < 1:
            raise ValueError(f"dim and max_length must be positive, got {dim}, {max_length}")
        rng = np.random.default_rng(seed)
        self.weight = Tensor(rng.normal(0.0, 0.02, size=(max_length, dim)))
        self.dim = dim
        self.max_length = max_length

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-2]
        if length > self.max_length:
            raise IndexError(
                f"sequence of {length} exceeds max_length {self.max_length}; "
                "a learned table has no row for it"
            )
        return x + take_rows(self.weight, np.arange(length))


def permutation_gap(layer, x: Tensor, order) -> float:
    """How much a layer's output changes when its input is shuffled.

    Runs the layer on ``x`` and on ``x[order]``, then un-shuffles the
    second result and compares. **Zero means the layer is permutation-
    equivariant** — it cannot tell one ordering from another, and a
    language model built from it could not distinguish "dog bites man"
    from "man bites dog".

    This is the measurement Day 11's second failed attempt ran into
    without naming.
    """
    order = np.asarray(order)
    inverse = np.argsort(order)
    straight = layer(x)
    straight = straight[0] if isinstance(straight, tuple) else straight
    shuffled = layer(Tensor(x.data[order]))
    shuffled = shuffled[0] if isinstance(shuffled, tuple) else shuffled
    return float(np.abs(straight.data - shuffled.data[inverse]).max())

class LayerNorm(Module):
    """Normalize each vector to zero mean and unit variance, then rescale.

        y = (x - mean) / sqrt(var + eps) * gamma + beta

    The statistics are taken **per position, over the feature axis** — not
    over the batch. That is the difference from BatchNorm and it is the
    reason transformers use this one: the result for a given token does
    not depend on which other tokens happen to share its batch, so
    training and single-sequence inference compute exactly the same thing.

    ``gamma`` starts at 1 and ``beta`` at 0, so the layer begins as pure
    normalization and learns its way out if it wants to. Starting ``gamma``
    anywhere else would apply a random rescale before training has any
    reason to.

    ``eps`` goes **inside** the square root. Outside it would not prevent
    a division by zero when the variance is zero — a constant input, which
    happens with padding — and the gradient would still be infinite.

    The output standard deviation is 0.999996 rather than exactly 1: eps
    sits inside the root and shrinks it very slightly. That is the price
    of never dividing by zero, and it is the right trade.

    >>> layer = LayerNorm(4)
    >>> out = layer(Tensor([[1.0, 2.0, 3.0, 4.0]]))
    >>> round(float(out.data.mean()), 9), round(float(out.data.std()), 4)
    (0.0, 1.0)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.dim = dim
        self.eps = eps
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape}")
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        variance = (centred ** 2).mean(axis=-1, keepdims=True)
        normalized = centred * ((variance + self.eps) ** -0.5)
        return normalized * self.gamma + self.beta


class Residual(Module):
    """``x + f(x)`` — the identity path that makes depth survivable.

    The gradient of a sum passes **unchanged** to both branches (Day 1),
    so ``d(x + f(x))/dx = 1 + f'(x)``. The ``1`` is a road straight from
    the loss to every earlier layer that no amount of depth can attenuate:
    even if ``f'`` is tiny at every layer, the product of ``(1 + small)``
    terms stays near 1, where a product of ``small`` terms does not.

    The demo measures both over twenty layers.
    """

    def __init__(self, inner: Module) -> None:
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        out = self.inner(x)
        out = out[0] if isinstance(out, tuple) else out
        if out.shape != x.shape:
            raise ValueError(
                f"a residual needs matching shapes, got {x.shape} and {out.shape}"
            )
        return x + out


class PreNormResidual(Module):
    """``x + f(norm(x))`` — normalize going in, add the raw input back.

    The arrangement every modern transformer uses. The residual path is
    then a clean identity from input to output with **nothing on it at
    all**, so a gradient reaches layer 1 from layer 40 without passing
    through a single normalization.

    Post-norm — ``norm(x + f(x))``, the original 2017 paper — puts a
    LayerNorm on that path at every layer. It trains, and it needs a
    learning-rate warmup to do so; pre-norm generally does not. The demo
    measures the gradient difference that explains why.
    """

    def __init__(self, dim: int, inner: Module, eps: float = 1e-5) -> None:
        self.norm = LayerNorm(dim, eps)
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        out = self.inner(self.norm(x))
        out = out[0] if isinstance(out, tuple) else out
        return x + out


class PostNormResidual(Module):
    """``norm(x + f(x))`` — the original ordering, kept for comparison."""

    def __init__(self, dim: int, inner: Module, eps: float = 1e-5) -> None:
        self.norm = LayerNorm(dim, eps)
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        out = self.inner(x)
        out = out[0] if isinstance(out, tuple) else out
        return self.norm(x + out)


def gradient_through_depth(depth: int, dim: int = 16, residual: bool = True,
                           normalize: bool = False, scale: float = 0.6,
                           seed: int = 0) -> float:
    """Mean |gradient| reaching the input of a stack ``depth`` layers deep.

    Each layer is ``tanh(x W)`` with weights deliberately scaled a little
    small, which is the situation Day 5 showed kills a deep signal. The
    question is whether the *gradient* survives the trip back.
    """
    rng = np.random.default_rng(seed)
    weights = [Tensor(rng.normal(0.0, scale / math.sqrt(dim), size=(dim, dim)))
               for _ in range(depth)]
    norms = [LayerNorm(dim) for _ in range(depth)] if normalize else None

    x = Tensor(np.random.default_rng(seed + 1).normal(size=(8, dim)))
    h = x
    for layer in range(depth):
        inner = h
        if normalize:
            inner = norms[layer](inner)
        transformed = (inner @ weights[layer]).tanh()
        h = h + transformed if residual else transformed
    h.sum().backward()
    return float(np.abs(x.grad).mean())

def gelu(x: Tensor) -> Tensor:
    """GELU, in the ``tanh`` approximation GPT-2 uses.

        0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))

    ReLU's hard zero discards a unit entirely; GELU keeps a little of the
    negative side, so a unit that is slightly wrong still receives
    gradient instead of switching off. The exact form uses the Gaussian
    CDF; this approximation is what the GPT-2 code shipped and what every
    reimplementation compares against.

    >>> round(float(gelu(Tensor([0.0])).data[0]), 6)
    0.0
    """
    inner = (x + (x ** 3) * 0.044715) * math.sqrt(2.0 / math.pi)
    return x * (inner.tanh() + 1.0) * 0.5


class FeedForward(Module):
    """Two linear layers with a nonlinearity, applied to each position alone.

    Attention moves information **between** positions; this moves it
    between *dimensions*, and it does so independently at every position
    — the same weights, applied separately, with no mixing across time.
    That is the division of labour in a transformer block, and it is why
    stacking them alternates the two.

    The hidden width is conventionally ``4 × dim``. That single constant
    is where most of the parameters go: ``8 d²`` here against ``4 d²`` for
    attention, so **two thirds of a transformer's weights are in the
    feed-forward layers**, which surprises anyone who assumes attention
    dominates. The demo counts them.

    >>> block = FeedForward(dim=8, seed=0)
    >>> block(Tensor(np.zeros((3, 8)))).shape
    (3, 8)
    """

    def __init__(self, dim: int, hidden: int | None = None,
                 activation: str = "gelu", seed: int | None = None) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if activation not in {"gelu", "relu", "tanh"}:
            raise ValueError(f"unknown activation {activation!r}")
        self.dim = dim
        self.hidden = hidden or 4 * dim
        self.activation = activation
        self.up = Linear(dim, self.hidden, seed=seed)
        self.down = Linear(self.hidden, dim, seed=None if seed is None else seed + 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.up(x)
        if self.activation == "gelu":
            hidden = gelu(hidden)
        elif self.activation == "relu":
            hidden = hidden.relu()
        else:
            hidden = hidden.tanh()
        return self.down(hidden)


class TransformerBlock(Module):
    """Pre-norm attention, then pre-norm feed-forward, each with a residual.

        x = x + attention(norm(x))
        x = x + feedforward(norm(x))

    Two sub-layers, two normalizations, two identity paths. Stacking these
    is the entire architecture — GPT-2 is twelve of them, GPT-3 is
    ninety-six, and nothing else changes.

    The **order** is not arbitrary. Attention gathers information from
    other positions; the feed-forward then processes what was gathered.
    Reversing them would process before gathering, which wastes the
    larger of the two layers on information the position already had.

    >>> block = TransformerBlock(dim=8, heads=2, seed=0)
    >>> block(Tensor(np.zeros((5, 8)))).shape
    (5, 8)
    """

    def __init__(self, dim: int, heads: int = 1, hidden: int | None = None,
                 causal: bool = True, seed: int | None = None) -> None:
        self.attention_norm = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, heads, causal=causal, seed=seed)
        self.feedforward_norm = LayerNorm(dim)
        self.feedforward = FeedForward(dim, hidden, seed=None if seed is None else seed + 10)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        attended, _ = self.attention(self.attention_norm(x))
        x = x + attended
        return x + self.feedforward(self.feedforward_norm(x))

    def attention_weights(self, x: Tensor) -> Tensor:
        """The weights this block's attention produces, for inspection."""
        return self.attention(self.attention_norm(x))[1]


def parameter_split(dim: int, heads: int = 1, hidden: int | None = None) -> dict:
    """Where a block's parameters actually live. Counts, not estimates."""
    block = TransformerBlock(dim=dim, heads=heads, hidden=hidden, seed=0)
    return {
        "attention": block.attention.parameter_count(),
        "feedforward": block.feedforward.parameter_count(),
        "layernorms": (block.attention_norm.parameter_count()
                       + block.feedforward_norm.parameter_count()),
        "total": block.parameter_count(),
    }

class CharTokenizer:
    """Every distinct character is a token. The smallest useful vocabulary.

    NLP-Lab spent a whole level on tokenization and Day 12 of it built
    BPE, which is what a real GPT uses. Characters are chosen here for a
    different reason: a vocabulary of forty means the output layer is
    forty wide, so a model small enough to train in this autograd can
    still produce recognisable text.

    >>> tokenizer = CharTokenizer("abcabc")
    >>> tokenizer.vocab_size
    3
    >>> tokenizer.decode(tokenizer.encode("cab"))
    'cab'
    """

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("cannot build a vocabulary from empty text")
        self.characters = sorted(set(text))
        self.to_id = {c: i for i, c in enumerate(self.characters)}
        self.to_char = {i: c for c, i in self.to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> np.ndarray:
        missing = set(text) - self.to_id.keys()
        if missing:
            raise KeyError(f"characters not in the vocabulary: {sorted(missing)!r}")
        return np.array([self.to_id[c] for c in text], dtype=int)

    def decode(self, ids) -> str:
        return "".join(self.to_char[int(i)] for i in np.asarray(ids).reshape(-1))


class GPT(Module):
    """Embeddings, positional encoding, N blocks, a final norm, a projection.

    The whole architecture, and every piece has already been built and
    gradient-checked on an earlier day:

        tokens -> embedding (Day 8)
               -> + positional encoding (Day 12)
               -> N x TransformerBlock (Day 14)
               -> LayerNorm (Day 13)
               -> Linear to vocabulary (Day 5)
               -> cross-entropy (Day 6)

    The **final LayerNorm** is easy to leave out and matters: without it
    the logits inherit whatever scale the residual stream has drifted to.
    Day 13 measured that stream growing with depth, and depth is the
    reason I first gave here — but measuring it says otherwise. On a
    fresh model, sixteen blocks inflate the logits 4x against one block's
    1.7x; on a two-block model, *training alone* takes the same ratio
    from 2.9x to 22.9x. Learned weights grow the stream, not depth.

    ``tie_weights`` reuses the embedding table as the output projection.
    The two have the same shape transposed, and the argument for sharing
    is that both answer the same question — how close is this vector to
    the vector for token *k*. It saves ``vocab × dim`` parameters, which
    on a real model is a tenth of the total. Day 5's parameter walk
    deduplicates by identity, so the shared tensor is updated once.

    >>> model = GPT(vocab_size=10, dim=8, blocks=1, heads=2, seed=0)
    >>> model(np.array([1, 2, 3])).shape
    (3, 10)
    """

    def __init__(self, vocab_size: int, dim: int = 32, blocks: int = 2,
                 heads: int = 2, max_length: int = 128, tie_weights: bool = True,
                 seed: int | None = None) -> None:
        if vocab_size < 1 or dim < 1 or blocks < 1:
            raise ValueError("vocab_size, dim and blocks must all be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_length = max_length
        self.tokens = Embedding(vocab_size, dim, seed=seed)
        self.positions = SinusoidalPositionalEncoding(dim, max_length)
        self.blocks = ModuleList([
            TransformerBlock(dim, heads, causal=True,
                             seed=None if seed is None else seed + 100 * (i + 1))
            for i in range(blocks)
        ])
        self.final_norm = LayerNorm(dim)
        self.tied = tie_weights
        self.head = None if tie_weights else Linear(dim, vocab_size, bias=False, seed=seed)

    def forward(self, ids) -> Tensor:
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError(f"expected a 1-D sequence of token ids, got shape {ids.shape}")
        if len(ids) > self.max_length:
            raise ValueError(f"sequence of {len(ids)} exceeds max_length {self.max_length}")
        h = self.positions(self.tokens(ids))
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h @ self.tokens.weight.T if self.tied else self.head(h)

    def loss(self, ids) -> Tensor:
        """Next-token cross-entropy over one sequence.

        Position ``t`` predicts token ``t+1``, so the logits are taken
        from all but the last position and the targets from all but the
        first. Getting this offset wrong by one is the classic way to
        train a model that predicts the token it was just given — and it
        produces a suspiciously low loss rather than an error.
        """
        ids = np.asarray(ids)
        if len(ids) < 2:
            raise ValueError("need at least two tokens to form a prediction")
        logits = self(ids[:-1])
        return cross_entropy(logits, ids[1:])


def sequence_batches(ids: np.ndarray, length: int, count: int, seed: int = 0):
    """Random windows of ``length + 1`` tokens, so each yields ``length``
    predictions."""
    if len(ids) < length + 1:
        raise ValueError(f"corpus of {len(ids)} is shorter than a window of {length + 1}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - length - 1, size=count)
    return [ids[start:start + length + 1] for start in starts]


def train_gpt(model: GPT, ids: np.ndarray, steps: int = 200, length: int = 32,
              batch: int = 8, lr: float = 0.01, seed: int = 0, report=None):
    """Train on random windows. Returns the loss history."""
    optimizer = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        windows = [ids[s:s + length + 1]
                   for s in rng.integers(0, len(ids) - length - 1, size=batch)]
        optimizer.zero_grad()
        total = 0.0
        for window in windows:
            loss = model.loss(window)
            loss.backward()
            total += float(loss.data)
        for parameter in model.parameters():
            parameter.grad = parameter.grad / batch
        optimizer.step()
        history.append(total / batch)
        if report and (step + 1) % report == 0:
            print(f"    step {step + 1:>4}: loss {history[-1]:.4f}")
    return history


#: A tiny corpus with real structure: repeated words, a fixed alphabet,
#: and enough regularity that a small model can learn something visible.
CORPUS = (
    "the cat sat on the mat. the dog sat on the log. "
    "the cat ran to the mat. the dog ran to the log. "
    "a cat and a dog sat on a mat. a dog and a cat ran to a log. "
    "the mat was flat and the log was round. "
    "the cat likes the mat and the dog likes the log. "
) * 12

def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by ``temperature`` before the softmax.

    Temperature sharpens or flattens the distribution without changing
    the *order* of the candidates. Below 1 the gaps widen and the model
    grows confident; above 1 they narrow and it grows adventurous. At
    exactly 0 the limit is the argmax, which is why greedy decoding is
    "temperature 0" rather than a separate algorithm.
    """
    if temperature < 0:
        raise ValueError(f"temperature must not be negative, got {temperature}")
    if temperature == 0:
        raise ValueError("temperature 0 is the argmax limit; sample_next handles it")
    return np.asarray(logits, dtype=float) / temperature


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax of a single row of logits, after temperature."""
    scaled = temperature_scale(logits, temperature)
    shifted = scaled - scaled.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` most likely tokens, renormalize, zero the rest.

    The tail of a softmax over a large vocabulary holds thousands of
    tokens that are individually absurd and collectively probable. Left
    in, one of them eventually gets sampled and the text derails. This
    is the bluntest fix: a fixed cutoff by rank.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k = min(k, probs.size)
    kept = np.zeros_like(probs)
    keep = np.argpartition(probs, -k)[-k:]
    kept[keep] = probs[keep]
    return kept / kept.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: the smallest set of tokens whose mass reaches ``p``.

    The objection to top-k is that ``k`` is fixed while the distribution
    is not. When the model is certain, 40 candidates admit 39 bad ones;
    when it is unsure, 40 may cut off a genuine option. Top-p adapts —
    it takes as many tokens as it takes to reach ``p`` of the mass, which
    is few in a confident position and many in an open one.

    The most likely token is always kept, so a distribution whose top
    entry already exceeds ``p`` degenerates to greedy rather than empty.
    """
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cut = int(np.searchsorted(cumulative, p)) + 1
    kept = np.zeros_like(probs)
    kept[order[:cut]] = probs[order[:cut]]
    return kept / kept.sum()


def sample_next(logits, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, rng=None) -> int:
    """One token id from one row of logits.

    ``temperature=0`` is greedy. The filters compose: temperature first,
    then top-k, then top-p, which is the order the reference
    implementations use and the only one where ``p`` means what it says.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if temperature == 0:
        return int(np.argmax(logits))
    probs = probabilities(logits, temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.choice(len(probs), p=probs))


def generate(model, tokenizer, prompt: str, length: int = 60,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, seed: int | None = None,
             context: int | None = None) -> str:
    """Extend ``prompt`` by ``length`` characters, one at a time.

    Every step is a full forward pass over the whole context, which is
    what makes generation expensive: producing ``n`` tokens costs ``n``
    forward passes, not one. Real implementations cache the keys and
    values of earlier positions; this one does not, and Day 17 measures
    what that costs.
    """
    rng = np.random.default_rng(seed)
    context = context or model.max_length
    ids = list(tokenizer.encode(prompt))
    for _ in range(length):
        logits = model(np.array(ids[-context:]))
        ids.append(sample_next(logits.data[-1], temperature, top_k, top_p, rng))
    return tokenizer.decode(ids)


def perplexity(model, ids, window: int = 64) -> float:
    """exp of the mean next-token negative log-likelihood.

    "How many characters is the model effectively choosing between at
    each step." A uniform model over a vocabulary of 20 scores exactly
    20; anything lower is knowledge. NLP-Lab's Day 11 built this for
    n-gram models and the definition does not change here — only what
    produces the probabilities.

    The corpus is walked in non-overlapping windows, so each token is
    predicted exactly once. The first token of each window has no
    predecessor inside it and is not scored, which slightly flatters a
    small window; the demo shows the size of that effect rather than
    picking a window and hoping.

    **The window must not exceed the window the model trained on.** A
    model trained on sequences of 32 has never seen position 32, and
    scoring it at window 64 measures positions it never learned rather
    than text it cannot predict. That is not a hypothetical: it is how
    this function first gave me 9.16 where 2.76 was correct. The demo
    tabulates both.
    """
    ids = np.asarray(ids)
    if len(ids) < 2:
        raise ValueError("need at least two tokens to score a prediction")
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            break
        logits = model(chunk[:-1]).data
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        total += -log_probs[np.arange(len(chunk) - 1), chunk[1:]].sum()
        count += len(chunk) - 1
    return float(np.exp(total / count))


def distinct_ratio(text: str, n: int = 4) -> float:
    """Distinct n-grams over total n-grams: 1.0 never repeats, 0.0 loops.

    The simplest number that catches the failure greedy decoding always
    produces. A model can have an excellent perplexity and still emit
    "the log the log the log" forever, because perplexity scores its
    predictions against *real* text and never looks at what it writes
    unprompted.
    """
    if len(text) < n:
        return 1.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams)


class CharNgram:
    """An add-k smoothed character n-gram model, as a baseline to beat.

    NLP-Lab spent Level 3 on these. The point of bringing one here is
    that a 25,000-parameter transformer trained for eleven seconds ought
    to justify itself against a model with no parameters at all — and on
    a corpus this small, the honest answer is in the demo.
    """

    def __init__(self, order: int = 3, k: float = 0.1) -> None:
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.order = order
        self.k = k
        self.counts: dict[tuple, np.ndarray] = {}
        self.vocab_size = 0

    def fit(self, ids, vocab_size: int) -> "CharNgram":
        ids = np.asarray(ids)
        self.vocab_size = vocab_size
        for position in range(len(ids)):
            history = tuple(ids[max(0, position - self.order + 1):position])
            history = (-1,) * (self.order - 1 - len(history)) + history
            row = self.counts.setdefault(history, np.zeros(vocab_size))
            row[ids[position]] += 1
        return self

    def distribution(self, history) -> np.ndarray:
        history = tuple(history)[-(self.order - 1):] if self.order > 1 else ()
        history = (-1,) * (self.order - 1 - len(history)) + history
        row = self.counts.get(history, np.zeros(self.vocab_size))
        smoothed = row + self.k
        return smoothed / smoothed.sum()

    def perplexity(self, ids) -> float:
        ids = np.asarray(ids)
        total = 0.0
        for position in range(1, len(ids)):
            probs = self.distribution(ids[:position])
            total += -math.log(probs[ids[position]])
        return float(math.exp(total / (len(ids) - 1)))


#: Sentences built from the training vocabulary in combinations the
#: training corpus never contains. The point of Day 16's central
#: measurement: a random split of a repeated corpus is not held out.
NOVEL_SENTENCES = (
    "the dog sat on the mat. the cat sat on the log. "
    "a cat likes a log and a dog likes a mat. "
    "the mat was round and the log was flat. "
    "the dog ran to the mat and the cat ran to the log. "
)

try:  # PyTorch is the only third-party import in this repo, and only here.
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the skip path
    torch = None
    F = None
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Every other day in this repo runs "
            "without it; this comparison cannot."
        )


class LeafStore:
    """Holds the torch copies of this repo's weights across rebuilds.

    Timing a forward pass that also *constructs* every parameter tensor
    measures allocation, not arithmetic, and charges PyTorch for work
    this repo's version does once at startup. The store builds the leaves
    on the first pass and hands back the same ones afterwards.
    """

    def __init__(self) -> None:
        self.items: list = []
        self._cursor = 0
        self._building = True

    def get(self, array, name, source):
        if self._building:
            tensor = torch.tensor(np.array(array, dtype=np.float64),
                                  requires_grad=True)
            self.items.append((name, source, tensor))
            return tensor
        tensor = self.items[self._cursor][2]
        self._cursor += 1
        return tensor

    def begin_pass(self) -> None:
        """Called at the start of every rebuild, so each pass hands the
        leaves out from the beginning again. Forgetting this is what made
        the first reuse work and the second raise IndexError."""
        self._cursor = 0

    def rewind(self) -> None:
        """Stop building; reuse what was built from here on."""
        self._building = False
        self._cursor = 0

    def zero(self) -> None:
        for _, _, tensor in self.items:
            tensor.grad = None


def _torch_block(h, block, index, store):
    """One TransformerBlock, rebuilt from torch's own primitives.

    Deliberately *not* a retyping of this repo's formulas: LayerNorm,
    GELU and attention come from ``torch.nn.functional``, so agreement
    means two independent implementations agree rather than one
    implementation agreeing with its own transcription.
    """
    prefix = f"blocks.{index}"
    attention = block.attention
    length, dim = h.shape[-2], block.dim
    head_dim = attention.head_dim

    gamma = store.get(block.attention_norm.gamma.data,
                      f"{prefix}.attention_norm.gamma", block.attention_norm.gamma)
    beta = store.get(block.attention_norm.beta.data,
                     f"{prefix}.attention_norm.beta", block.attention_norm.beta)
    normed = F.layer_norm(h, (dim,), gamma, beta, block.attention_norm.eps)

    projections = {}
    for role in ("query", "key", "value", "out"):
        layer = getattr(attention, role)
        projections[role] = store.get(layer.weight.data,
                                      f"{prefix}.attention.{role}.weight", layer.weight)

    def heads(x):
        return x.reshape(length, attention.heads, head_dim).permute(1, 0, 2)

    attended = F.scaled_dot_product_attention(
        heads(normed @ projections["query"]),
        heads(normed @ projections["key"]),
        heads(normed @ projections["value"]),
        is_causal=attention.causal,
    )
    merged = attended.permute(1, 0, 2).reshape(length, dim)
    h = h + merged @ projections["out"]

    gamma2 = store.get(block.feedforward_norm.gamma.data,
                       f"{prefix}.feedforward_norm.gamma", block.feedforward_norm.gamma)
    beta2 = store.get(block.feedforward_norm.beta.data,
                      f"{prefix}.feedforward_norm.beta", block.feedforward_norm.beta)
    normed2 = F.layer_norm(h, (dim,), gamma2, beta2, block.feedforward_norm.eps)

    up_w = store.get(block.feedforward.up.weight.data,
                     f"{prefix}.feedforward.up.weight", block.feedforward.up.weight)
    up_b = store.get(block.feedforward.up.bias.data,
                     f"{prefix}.feedforward.up.bias", block.feedforward.up.bias)
    down_w = store.get(block.feedforward.down.weight.data,
                       f"{prefix}.feedforward.down.weight", block.feedforward.down.weight)
    down_b = store.get(block.feedforward.down.bias.data,
                       f"{prefix}.feedforward.down.bias", block.feedforward.down.bias)
    hidden = F.gelu(normed2 @ up_w + up_b, approximate="tanh")
    return h + hidden @ down_w + down_b


def torch_rebuild(model, ids, store: "LeafStore | None" = None):
    """Recompute ``GPT.loss`` in PyTorch from the same weights.

    Returns ``(logits, loss, store)``. Everything runs in float64, so a
    disagreement is a difference in the *derivative* rather than in the
    arithmetic - which is the point of the exercise.
    """
    require_torch()
    ids = np.asarray(ids)
    store = LeafStore() if store is None else store
    store.begin_pass()
    inputs = torch.tensor(ids[:-1].astype(np.int64))
    targets = torch.tensor(ids[1:].astype(np.int64))

    table = store.get(model.tokens.weight.data, "tokens.weight", model.tokens.weight)
    h = table[inputs] + torch.tensor(model.positions.table[:len(inputs)])
    for index, block in enumerate(model.blocks):
        h = _torch_block(h, block, index, store)

    gamma = store.get(model.final_norm.gamma.data, "final_norm.gamma",
                      model.final_norm.gamma)
    beta = store.get(model.final_norm.beta.data, "final_norm.beta",
                     model.final_norm.beta)
    h = F.layer_norm(h, (model.dim,), gamma, beta, model.final_norm.eps)

    if model.tied:
        logits = h @ table.T
    else:
        logits = h @ store.get(model.head.weight.data, "head.weight",
                               model.head.weight)
    return logits, F.cross_entropy(logits, targets), store


def forward_agreement(model, ids) -> float:
    """Largest absolute difference between the two forward passes.

    Checked **before** any gradient is compared. If the forwards differ,
    the two graphs are not the same computation and comparing their
    derivatives would prove nothing - a failure here means the torch
    rebuild is wrong, not that the autograd is.
    """
    logits, _, _ = torch_rebuild(model, ids)
    mine = model(np.asarray(ids)[:-1]).data
    return float(np.abs(mine - logits.detach().numpy()).max())


def gradient_agreement(model, ids) -> dict:
    """Per-parameter relative difference between this autograd and torch.

    The number reported is ``max|mine - theirs| / max|theirs|`` - relative
    to the gradient's own scale, because an absolute gap of 1e-9 means
    something very different on a gradient of 1e-8 than on one of 1e3.
    """
    _, loss, store = torch_rebuild(model, ids)
    loss.backward()
    for parameter in model.parameters():
        parameter.grad = np.zeros_like(parameter.data)
    model.loss(np.asarray(ids)).backward()

    report = {}
    for name, source, tensor in store.items:
        reference = tensor.grad.numpy()
        scale = max(float(np.abs(reference).max()), 1e-12)
        report[name] = float(np.abs(source.grad - reference).max() / scale)
    return report


def time_forward_and_backward(model, ids, repeats: int = 10,
                              warmup: int = 2) -> dict:
    """Seconds for a forward and for a backward pass of this repo's autograd.

    Both are timed **inside the same iteration**, because timing them in
    separate loops and subtracting gave a negative backward time on the
    first attempt - warm-up drift between the two loops was larger than
    the quantity being measured.
    """
    ids = np.asarray(ids)
    for _ in range(warmup):
        model.loss(ids).backward()

    forward = backward = 0.0
    for _ in range(repeats):
        for parameter in model.parameters():
            parameter.grad = np.zeros_like(parameter.data)
        start = time.perf_counter()
        loss = model.loss(ids)
        middle = time.perf_counter()
        loss.backward()
        end = time.perf_counter()
        forward += middle - start
        backward += end - middle
    return {"forward": forward / repeats, "backward": backward / repeats,
            "total": (forward + backward) / repeats}


def torch_time_forward_and_backward(model, ids, repeats: int = 10,
                                    warmup: int = 2) -> dict:
    """The same measurement through PyTorch, on the same weights.

    The leaves are built once and reused, so this times arithmetic rather
    than tensor construction.
    """
    require_torch()
    ids = np.asarray(ids)
    _, loss, store = torch_rebuild(model, ids)
    loss.backward()
    store.rewind()
    for _ in range(warmup):
        store.zero()
        _, loss, _ = torch_rebuild(model, ids, store)
        loss.backward()

    forward = backward = 0.0
    for _ in range(repeats):
        store.zero()
        start = time.perf_counter()
        _, loss, _ = torch_rebuild(model, ids, store)
        middle = time.perf_counter()
        loss.backward()
        end = time.perf_counter()
        forward += middle - start
        backward += end - middle
    return {"forward": forward / repeats, "backward": backward / repeats,
            "total": (forward + backward) / repeats}


def cost_by_length(dim: int, blocks: int, heads: int, lengths, vocab: int = 20,
                   repeats: int = 3) -> list:
    """Seconds per forward pass as the sequence grows.

    Attention is ``O(T²)`` in the sequence length; every other part is
    ``O(T)``. So the measured exponent sits between 1 and 2 and climbs
    towards 2 as attention comes to dominate - which is a claim about
    *where* the crossover is, and depends on the width.
    """
    model = GPT(vocab_size=vocab, dim=dim, blocks=blocks, heads=heads,
                max_length=max(lengths), seed=0)
    rows = []
    for length in lengths:
        ids = np.arange(length + 1) % vocab
        model(ids[:-1])  # warm up this length
        seconds = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            model(ids[:-1])
            seconds += time.perf_counter() - start
        rows.append((length, seconds / repeats))
    return rows


def scaling_exponent(rows) -> float:
    """Slope of log(time) against log(length): 1 is linear, 2 quadratic."""
    lengths = np.log(np.array([row[0] for row in rows], dtype=float))
    seconds = np.log(np.array([row[1] for row in rows], dtype=float))
    return float(np.polyfit(lengths, seconds, 1)[0])


def generation_cost(model, length: int, vocab: int = 20) -> dict:
    """What generating ``length`` tokens costs without a KV cache.

    Every step re-runs the whole prefix, so producing ``n`` tokens
    processes ``1 + 2 + ... + n = n(n+1)/2`` positions where a cached
    implementation processes ``n``. Nothing in this repo caches, and the
    ratio is what that decision costs.
    """
    ids = [0]
    positions = 0
    start = time.perf_counter()
    for _ in range(length):
        window = np.array(ids[-model.max_length:])
        logits = model(window)
        positions += len(window)
        ids.append(int(np.argmax(logits.data[-1])) % vocab)
    return {"seconds": time.perf_counter() - start,
            "positions_processed": positions,
            "positions_with_a_cache": length,
            "waste": positions / length}


def graph_size(model, ids) -> dict:
    """How many nodes one training step builds, and what that costs.

    The profiler's top entry for a step is ``numpy.zeros_like``, not a
    matrix multiply. Every ``Tensor`` allocates a zero gradient array
    when it is constructed, and a forward pass constructs one per
    intermediate operation - so a small model spends more time allocating
    gradient buffers than doing the arithmetic they hold.

    The graph is its own instrument here: Day 1's ``topological_order``
    already enumerates exactly the tensors that were built, so counting
    them needs no counter bolted onto ``Tensor``.
    """
    loss = model.loss(np.asarray(ids))
    nodes = loss.topological_order()
    return {"nodes": len(nodes), "parameters": len(model.parameters())}


def hot_spots(model, ids, count: int = 6, steps: int = 3) -> list:
    """The functions a training step actually spends its time in.

    Returns ``(label, seconds, calls)`` sorted by self time. The answer is
    not the one most people would guess, which is why it is measured
    rather than asserted.
    """
    ids = np.asarray(ids)
    model.loss(ids).backward()  # warm up

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(steps):
        for parameter in model.parameters():
            parameter.grad = np.zeros_like(parameter.data)
        model.loss(ids).backward()
    profiler.disable()

    stats = pstats.Stats(profiler)
    rows = []
    for (path, line, name), entry in stats.stats.items():
        calls, _, tottime = entry[0], entry[1], entry[2]
        label = f"{pathlib.Path(path).name}:{name}" if path != "~" else name
        rows.append((label, tottime, calls))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:count]


def pairwise_slopes(rows) -> list:
    """Local log-log slope between consecutive lengths.

    A single fitted exponent hides the interesting part: the slope is
    near 1 while per-node Python overhead dominates and climbs towards 2
    as attention's ``T²`` term takes over. Reporting one number for the
    whole range would average those two regimes into something that
    describes neither.
    """
    slopes = []
    for before, after in zip(rows, rows[1:]):
        ratio = math.log(after[0] / before[0])
        slopes.append((before[0], after[0],
                       math.log(after[1] / before[1]) / ratio))
    return slopes


def attention_work(model, length: int) -> int:
    """Entries in every attention score matrix of one forward pass.

    The quadratic term, **counted rather than timed**. Each block holds
    ``heads`` score matrices of ``length × length``, so this is
    ``blocks × heads × T²`` and doubling the length must quadruple it
    exactly.

    This function exists because the timed version could not carry the
    claim. A fitted log-log exponent over wall-clock measurements came
    out between 0.79 and 1.23 across six consecutive local runs, and CI
    - a shared runner, timing sub-millisecond work - produced 0.38 for
    computation that is provably superlinear. Widening the tolerance
    until such a test passes does not make it a measurement of anything.
    Timing belongs in the demo, where it is reported; the assertion
    belongs on a quantity that does not depend on what else the machine
    is doing.
    """
    ids = np.arange(length) % model.vocab_size
    hidden = model.positions(model.tokens(ids))
    total = 0
    for block in model.blocks:
        total += block.attention_weights(hidden).data.size
        hidden = block(hidden)
    return total

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tokenizer = CharTokenizer(CORPUS)
    ids = tokenizer.encode(CORPUS[:65])
    model = GPT(vocab_size=tokenizer.vocab_size, dim=32, blocks=2, heads=4,
                max_length=1024, seed=0)
    print(f"model: dim 32, 2 blocks, 4 heads, {model.parameter_count():,} parameters")
    print(f"sixteen days of hand-derived calculus, {len(ids) - 1} positions of input")

    print()
    print("=" * 66)
    print("does it agree with PyTorch?")
    print("=" * 66)
    if not TORCH_AVAILABLE:
        print("  PyTorch is not installed, so this section did not run.")
        print("  Everything else in this repo runs without it. Install with")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  and run again for the comparison this day exists for.")
    else:
        print(f"  torch {torch.__version__}, float64, CPU")
        print()
        print(f"  forward pass, largest absolute difference: "
              f"{forward_agreement(model, ids):.2e}")
        print("  checked first: if the forwards disagree the two graphs are not")
        print("  the same computation, and comparing derivatives proves nothing")
        print()
        report = gradient_agreement(model, ids)
        print(f"  gradients, relative difference per parameter "
              f"({len(report)} tensors):")
        for name, difference in sorted(report.items(),
                                       key=lambda item: -item[1])[:6]:
            print(f"    {difference:.2e}   {name}")
        print(f"    ...")
        print(f"  worst of all {len(report)}: {max(report.values()):.2e}")
        print()
        print("  That is float64 round-off, roughly 1e-16 amplified by a few")
        print("  hundred operations. Every derivative in this repo was worked")
        print("  out by hand and checked against finite differences; this is")
        print("  the first time any of it has been compared with an")
        print("  implementation written by someone else, and the LayerNorm,")
        print("  GELU, attention and cross-entropy on the other side are")
        print("  torch's own, not mine retyped.")

    print()
    print("=" * 66)
    print("what it costs")
    print("=" * 66)
    mine = time_forward_and_backward(model, ids, repeats=60)
    print(f"  {'':<10}{'forward':>10}{'backward':>10}{'total':>10}")
    print(f"  {'this repo':<10}{mine['forward'] * 1000:>9.2f}m"
          f"{mine['backward'] * 1000:>9.2f}m{mine['total'] * 1000:>9.2f}m")
    if TORCH_AVAILABLE:
        theirs = torch_time_forward_and_backward(model, ids, repeats=60)
        print(f"  {'pytorch':<10}{theirs['forward'] * 1000:>9.2f}m"
              f"{theirs['backward'] * 1000:>9.2f}m{theirs['total'] * 1000:>9.2f}m")
        print()
        print(f"  slower overall by {mine['total'] / theirs['total']:.1f}x, which is")
        print("  the expected half of this. The unexpected half is the split:")
        print(f"    backward / forward, this repo {mine['backward'] / mine['forward']:.2f}")
        print(f"    backward / forward, pytorch   {theirs['backward'] / theirs['forward']:.2f}")
        print()
        print("  I expected these to match, on the reasoning that reverse mode")
        print("  makes a backward pass a small constant times the forward one")
        print("  whatever the implementation. They do not, and the gap says")
        print("  where each one's time goes. PyTorch's backward costs about")
        print("  1.5x its forward, which is roughly the FLOP ratio - a backward")
        print("  pass computes two gradients per matmul where the forward")
        print("  computed one product. This repo's backward costs LESS than")
        print("  its forward, because the forward is where the graph is built:")
        print("  220 Tensor objects, each allocating a gradient array, each")
        print("  closing over a Python function. The backward pass then walks")
        print("  a list that already exists. The arithmetic follows the same")
        print("  ratio in both; the Python object churn does not, and it is")
        print("  charged entirely to the forward.")

    print()
    print("where the time actually goes:")
    for label, seconds, calls in hot_spots(model, ids):
        print(f"  {seconds * 1000:7.2f} ms  {calls:>6} calls  {label}")
    size = graph_size(model, ids)
    print(f"  one step builds {size['nodes']} tensors for {size['parameters']} parameters")
    print()
    print("  The matmuls are there, as expected. What is not obvious is how")
    print("  much company they keep: hundreds of calls to zeros_like and to")
    print("  Tensor.__init__, for a model with 27 parameter tensors. Every")
    print("  Tensor allocates a zero gradient array when it is constructed,")
    print("  and the forward pass constructs one per operation - so a large")
    print("  share of a step at this size is spent making gradient buffers")
    print("  rather than filling them. That is a direct consequence of Day 1's")
    print("  design, where every node owns its gradient; a framework pools and")
    print("  reuses those buffers. At a realistic model size the matmuls would")
    print("  dominate and this overhead would vanish into them.")

    print()
    print("=" * 66)
    print("the two costs that are architectural, not implementation detail")
    print("=" * 66)
    rows = cost_by_length(32, 1, 4, [64, 128, 256, 512, 1024], repeats=2)
    print("  forward pass against sequence length:")
    for length, seconds in rows:
        print(f"    {length:>5}: {seconds * 1000:8.1f} ms")
    print("  local slope in log-log, 1 is linear and 2 is quadratic:")
    for before, after, slope in pairwise_slopes(rows):
        print(f"    {before:>4} -> {after:<5} {slope:.2f}")
    print()
    print("  the slope climbs. At short lengths the per-node Python overhead")
    print("  above dominates and the curve looks almost linear; attention's")
    print("  T-squared term only takes over further out. A single fitted")
    print("  exponent over the whole range would average two regimes into a")
    print("  number describing neither, which is why the slopes are local.")

    print()
    generation = generation_cost(model, 60)
    print(f"  generating 60 tokens took {generation['seconds']:.2f} s and processed")
    print(f"  {generation['positions_processed']:,} positions where a cached "
          f"implementation processes {generation['positions_with_a_cache']}")
    print(f"  - {generation['waste']:.1f}x more work, because nothing here caches keys")
    print("  and values. Every step re-runs the entire prefix from scratch. This")
    print("  is the single largest thing missing from this repo, and it is an")
    print("  engineering absence rather than a gap in the mathematics.")

    print()
    print("=" * 66)
    print("the honest account")
    print("=" * 66)
    for line in HONEST_ACCOUNT.strip().split("\n"):
        print(f"  {line}" if line else "")


#: The writeup this level exists for. Kept as data so the README and the
#: demo cannot drift apart.
HONEST_ACCOUNT = """
What this repo demonstrates, and what it does not.

It demonstrates that the mathematics is right. Every derivative was
worked out by hand, checked against finite differences the day it was
written, and today checked against PyTorch: agreement to 1e-15 relative
across every parameter tensor of a full GPT. Nothing was copied from a
reference implementation, and the parts that were wrong were found by
measurement rather than by reading someone else's code.

It does not demonstrate that the code is fast, and it was never going to.
On this machine it is several times slower than PyTorch on the same
arithmetic in the same precision - the run above prints the ratio - and
that gap would widen on a GPU, at a realistic model size, and against a
KV cache the generation loop here does not have.

The interesting part is which of those two facts is the hard one. The
calculus is the part a person can derive, verify and be sure of. The
performance is the part that takes a compiler team, and buying it does
not require understanding anything derived here.

A good many claims in this repo were wrong before they were measured,
and each is documented where it was made rather than quietly corrected:
backward() silently clearing gradients, which made zero_grad() dead code
and contradicted Day 1's own README (found on Day 7); Adam's bias
correction described backwards (Day 7); a gradient probe read the wrong
way round (Day 9); a capacity demo a single head solved by memorising
(Day 11); a permutation-equivariance claim the measurement contradicted
(Day 12); depth credited for what training does to the residual stream
(Day 15); a causality probe defeated by weight tying (Day 15); a
held-out split that was a copy of the training text (Day 16); a
perplexity measured past the window the model trained on (Day 16); and,
on this very day, a backward pass timed at a negative number of
seconds.

That last one is worth stating plainly. On 2,940 characters, a
count-based trigram scores a better perplexity than this model and
takes no training at all. The transformer's advantage is long-range
structure, and this corpus has none. A repo that reported only the
flattering half of that would be teaching the wrong lesson.
"""


if __name__ == "__main__":
    main()
