"""Day 7 — SGD, Adam, and a training loop that learns something.

Everything needed to train has existed since Day 6: a model with
parameters, a loss, and a gradient for every parameter. This day adds the
four lines that use them, and the result is the first thing in this repo
that actually learns.

The optimizer never sees the model. It is handed a list of tensors by
``Module.parameters()`` and updates them in place, which is why Day 5
spent its effort on making that list complete.

Two optimizers, because the difference between them is worth measuring
rather than reading about. **SGD** subtracts the gradient. **Adam** keeps
a running estimate of each parameter's gradient magnitude and divides by
it, so a parameter whose gradients are habitually a hundred times larger
gets a hundredth of the step — which is why badly scaled inputs barely
slow it down and stop SGD almost completely.

Adam's bias correction gets its own measurement, because the usual
description of it is backwards.
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

if __name__ == "__main__":
    print("the first thing in this repo that learns:\n")

    x, y = xor_dataset(400, seed=0)
    net = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
    optimizer = Adam(net.parameters(), lr=0.05)
    history = train(net, x, y, optimizer, steps=300)
    print(f"  XOR, {net.parameter_count()} parameters, 400 points")
    print(f"  loss {history[0]:.4f} -> {history[-1]:.4f}   "
          f"(chance is log 2 = {math.log(2):.4f})")
    print(f"  accuracy {accuracy(net(x), y):.1%}")
    print("  no straight line separates XOR, so this is a genuinely nonlinear fit")

    print("\na linear model provably cannot do it:")
    linear_only = Sequential(Linear(2, 2, seed=0))
    linear_history = train(linear_only, x, y, Adam(linear_only.parameters(), lr=0.05), 300)
    print(f"  loss {linear_history[0]:.4f} -> {linear_history[-1]:.4f}   "
          f"accuracy {accuracy(linear_only(x), y):.1%}")
    print("  it sits at chance, which is the correct answer for a line on XOR")

    print("\nAdam's bias correction, measured at step 1:")
    for corrected in (True, False):
        p = Tensor([0.0])
        p.grad = np.array([1.0])
        optimizer = Adam([p], lr=1.0, bias_correction=corrected)
        optimizer.step()
        label = "corrected" if corrected else "uncorrected"
        print(f"  {label:<12} first step = {abs(p.data.item()):.4f}")
    ratio = (1 - 0.9) / math.sqrt(1 - 0.999)
    print(f"  uncorrected is (1-b1)/sqrt(1-b2) = {ratio:.2f}x too LARGE, not too small -")
    print("  the opposite of what the phrase 'bias toward zero' suggests")

    print("\n  and it fades as t grows:")
    print(f"  {'step':>6}{'corrected':>12}{'uncorrected':>14}")
    for target_step in (1, 2, 5, 20, 100):
        sizes = []
        for corrected in (True, False):
            p = Tensor([0.0])
            optimizer = Adam([p], lr=1.0, bias_correction=corrected)
            for _ in range(target_step):
                p.grad = np.array([1.0])
                optimizer.step()
            sizes.append(abs(p.data.item()) / target_step)
        print(f"  {target_step:>6}{sizes[0]:>12.4f}{sizes[1]:>14.4f}")

    print("\nbadly scaled features: where Adam earns its keep")
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(300, 2))
    raw[:, 1] *= 100.0            # one feature a hundred times larger
    labels = (raw[:, 0] * 1.0 + raw[:, 1] * 0.01 > 0).astype(int)
    scaled_x = Tensor(raw)
    print(f"  {'optimizer':<22}{'final loss':>12}{'accuracy':>11}")
    for label, make in (
        ("SGD lr=1e-4", lambda p: SGD(p, lr=1e-4)),
        ("SGD lr=1e-4 + mom .9", lambda p: SGD(p, lr=1e-4, momentum=0.9)),
        ("Adam lr=0.05", lambda p: Adam(p, lr=0.05)),
    ):
        model = Sequential(Linear(2, 8, seed=0), Tanh(), Linear(8, 2, seed=1))
        losses = train(model, scaled_x, labels, make(model.parameters()), 400)
        print(f"  {label:<22}{losses[-1]:>12.4f}{accuracy(model(scaled_x), labels):>11.1%}")
    print("  Adam divides each parameter's step by its own gradient scale, so a")
    print("  feature 100x larger does not force a 100x smaller learning rate")

    print()
    print("learning rate: too large stops being gradient descent")
    print(f"  {'lr':>9}{'final':>10}{'worst seen':>13}{'steps worse':>13}")
    for lr in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        losses = train(model, x, y, SGD(model.parameters(), lr=lr), 200)
        uphill = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        print(f"  {lr:>9.2f}{max(losses[-1], 0.0) + 0.0:>10.4f}{max(losses):>13.4f}{uphill:>13}")
    print("  note what does NOT happen: nothing reaches inf or nan. tanh bounds every")
    print("  activation, so an absurd step cannot blow the network up - it only")
    print("  teleports the parameters somewhere arbitrary. At lr=100 the loss goes")
    print("  uphill on many steps, which is no longer descent in any sense; landing")
    print("  somewhere low anyway is luck, not learning.")

    print("\nforgetting zero_grad does not crash, it silently changes the model:")
    for clear in (True, False):
        model = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
        optimizer = SGD(model.parameters(), lr=0.05)
        for _ in range(100):
            loss = cross_entropy(model(x), y)
            if clear:
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        label = "with zero_grad" if clear else "without"
        print(f"  {label:<18} final loss {float(loss.data):.4f}  "
              f"accuracy {accuracy(model(x), y):.1%}")
    print("  gradients accumulate by design (Day 1), so step t applies the sum of")
    print("  every gradient so far - an effective learning rate that grows with the")
    print("  step count. Here that happened to help, which is exactly why it is")
    print("  dangerous: the bug can present as a better result.")
