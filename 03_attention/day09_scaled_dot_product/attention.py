"""Day 9 — Scaled dot-product attention.

Every layer so far applies weights that were learned once and then held
fixed: a ``Linear`` treats the thousandth example exactly like the first.
Attention is the first operation where **the coefficients are computed
from the input itself**, freshly, for every position of every example.

    attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Three matrices in, one out. ``Q`` asks, ``K`` advertises, ``V`` carries
the content, and the softmax turns the match between question and
advertisement into a set of mixing weights that sum to one.

The mechanism itself has **no parameters**. Everything learned lives in
the three projections that produce Q, K and V from the input.

The one constant in the formula is ``√d_k``, and this day measures why it
is there rather than quoting the paper. Without it the scores spread out
in proportion to ``√d_k``, the softmax saturates into a hard lookup, and
the gradient reaching the query and key projections collapses by orders
of magnitude.
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

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("attention computes its own weights from the input.")
    print("every layer so far applied weights that were learned and then fixed.")
    print()

    q = Tensor([[1.0, 0.0], [0.0, 1.0]])
    k = Tensor([[1.0, 0.0], [0.0, 1.0]])
    v = Tensor([[10.0, 0.0], [0.0, 20.0]])
    out, weights = attention(q, k, v)
    print("  a tiny example, query i matching key i:")
    print(f"    weights {weights.data.round(3).tolist()}")
    print(f"    output  {out.data.round(3).tolist()}")
    print(f"    rows sum to 1: {np.allclose(weights.data.sum(axis=-1), 1.0)}")

    print()
    print("why divide by sqrt(d_k) - the score spread grows with dimension:")
    print(f"  {'d_k':>6}{'sd unscaled':>14}{'sd scaled':>12}{'sqrt(d_k)':>12}")
    for d_k in (4, 16, 64, 256, 1024):
        query = Tensor(rng.normal(size=(64, d_k)))
        key = Tensor(rng.normal(size=(64, d_k)))
        raw = attention_scores(query, key, scale=False).data
        scaled = attention_scores(query, key, scale=True).data
        print(f"  {d_k:>6}{raw.std():>14.2f}{scaled.std():>12.2f}{math.sqrt(d_k):>12.2f}")
    print("  the unscaled column tracks sqrt(d_k) exactly, which is the whole")
    print("  reason the constant is sqrt(d_k) and not something tuned")

    print()
    print("what that does to the attention distribution (64 positions):")
    print(f"  {'d_k':>6}{'entropy unscaled':>19}{'entropy scaled':>17}{'uniform':>10}")
    uniform = math.log(64)
    for d_k in (4, 16, 64, 256, 1024):
        query = Tensor(rng.normal(size=(64, d_k)))
        key = Tensor(rng.normal(size=(64, d_k)))
        value = Tensor(rng.normal(size=(64, d_k)))
        _, unscaled_w = attention(query, key, value, scale=False)
        _, scaled_w = attention(query, key, value, scale=True)
        print(f"  {d_k:>6}{attention_entropy(unscaled_w).mean():>19.4f}"
              f"{attention_entropy(scaled_w).mean():>17.4f}{uniform:>10.4f}")
    print("  unscaled, attention collapses onto one position - entropy near 0.")
    print("  that is a hard lookup, not a weighted average.")

    print()
    print("and the softmax stops responding, which is the actual claim:")
    print(f"  {'d_k':>6}{'sensitivity unscaled':>23}{'scaled':>10}")
    for d_k in (4, 16, 64, 256, 1024):
        query = Tensor(rng.normal(size=(32, d_k)))
        key = Tensor(rng.normal(size=(32, d_k)))
        value = Tensor(rng.normal(size=(32, d_k)))
        row = []
        for use_scale in (False, True):
            _, w = attention(query, key, value, scale=use_scale)
            probabilities = np.asarray(w.data)
            row.append(float(np.mean(np.sum(probabilities * (1 - probabilities), axis=-1))))
        print(f"  {d_k:>6}{row[0]:>23.5f}{row[1]:>10.5f}")
    print("  this is sum_j p_j(1 - p_j), the diagonal of the softmax Jacobian -")
    print("  how much the weights move when a score moves. Unscaled it falls to")
    print("  0.07 at d_k=1024, a 12x collapse: the scores can change and the")
    print("  output barely will.")
    print()
    print("  a note on what NOT to measure: |d out / d query| is *larger* without")
    print("  scaling, because the 1/sqrt(d_k) factor multiplies that gradient")
    print("  directly. It conflates the scale with the saturation it is meant to")
    print("  detect, and reads exactly backwards. The Jacobian above does not.")

    print()
    print("self-attention: Q, K and V are three views of one input")
    layer = SelfAttention(dim=8, seed=0)
    x = Tensor(rng.normal(size=(5, 8)))
    out, weights = layer(x)
    print(f"  input {x.shape} -> output {out.shape}, weights {weights.shape}")
    print(f"  {layer.parameter_count()} parameters, all in the three projections")
    print("  the attention mechanism itself has none")

    print()
    print("every position can see every other one (Day 10 removes that):")
    print(f"  weight row 0: {weights.data[0].round(3).tolist()}")
    print(f"  position 0 attends to positions {list(range(weights.shape[1]))}")
    print("  including 1, 2, 3, 4 - which is fine here and fatal for a language model")

    print()
    print("checked against finite differences:")
    for name, function, inputs in (
        ("scores", lambda a, b: attention_scores(a, b).sum(),
         [rng.normal(size=(3, 4)), rng.normal(size=(5, 4))]),
        ("full attention", lambda a, b, c: attention(a, b, c)[0].sum(),
         [rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 4))]),
        ("attention squared", lambda a, b, c: (attention(a, b, c)[0] ** 2).sum(),
         [rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 4))]),
        ("self-attention", lambda x, wq, wk, wv: attention(x @ wq, x @ wk, x @ wv)[0].sum(),
         [rng.normal(size=(4, 6)), rng.normal(size=(6, 6)),
          rng.normal(size=(6, 6)), rng.normal(size=(6, 6))]),
    ):
        check_gradient(function, inputs)
        print(f"  {name:<20} ok")
