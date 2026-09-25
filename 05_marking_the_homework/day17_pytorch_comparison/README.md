# Day 17 · PyTorch Comparison, Profiling, and the Honest Writeup

> Sixteen days of derivatives worked out by hand and checked against finite differences. Today they are checked, for the first time, against an implementation written by somebody else.

```
forward pass, largest absolute difference: 1.67e-16

gradients, relative difference per parameter (27 tensors):
  2.71e-15   blocks.1.attention.query.weight
  2.42e-15   blocks.1.attention.key.weight
  1.25e-15   blocks.0.feedforward_norm.beta
  1.25e-15   blocks.0.attention.query.weight
  ...
worst of all 27: 2.71e-15
```

That is float64 round-off — about `1e-16` amplified by a few hundred operations. The whole repo, from Day 1's scalar `Value` to a trained GPT, computes the same derivatives PyTorch does.

Two details make this a real check rather than a ceremony:

**The forward is compared first.** If the two forward passes disagreed, the graphs would not be the same computation and comparing their derivatives would prove nothing. A failure there means the torch rebuild is wrong, not the autograd.

**The other side is torch's own code.** The comparison uses `F.layer_norm`, `F.gelu`, `F.scaled_dot_product_attention` and `F.cross_entropy` — not this repo's formulas retyped in torch, which would only prove the transcription was faithful. And a test deliberately perturbs a parameter to confirm the comparison *can* fail.

## What it costs

```
             forward   backward     total
this repo     6.13ms     3.80ms     9.93ms
pytorch       1.68ms     2.77ms     4.45ms
```

Slower, which is expected. The unexpected part is the split:

```
backward / forward, this repo  0.62
backward / forward, pytorch    1.64
```

I expected these to match, reasoning that reverse mode makes the backward pass a small constant times the forward one regardless of implementation. **They do not**, and the gap says where each one's time goes.

PyTorch's backward costs about 1.6× its forward, roughly the FLOP ratio — a backward pass computes two gradients per matmul where the forward computed one product. This repo's backward costs *less* than its forward, because the forward is where the graph gets built: 220 `Tensor` objects, each allocating a gradient array and closing over a Python function. The backward pass then walks a list that already exists.

The arithmetic follows the same ratio in both. The Python object churn does not, and it is charged entirely to the forward.

```
   5.37 ms      51 calls  _backward
   3.62 ms      51 calls  __matmul__
   2.78 ms     663 calls  numpy zeros_like
   1.49 ms     579 calls  Tensor.__init__

one step builds 220 tensors for 27 parameters
```

663 allocations for a model with 27 parameter tensors. Every `Tensor` zeroes a gradient array when constructed — a direct consequence of Day 1's design, where each node owns its gradient. A framework pools and reuses those buffers. At a realistic model size the matmuls would swallow this entirely.

## Two costs that are architectural, not implementation detail

```
forward pass against sequence length      local log-log slope
     64:      5.0 ms                        64 -> 128   1.25
    128:     11.8 ms                       128 -> 256   1.40
    256:     31.1 ms                       256 -> 512   1.25
    512:     73.9 ms                       512 -> 1024  1.95
   1024:    286.0 ms
```

The slope climbs toward 2. At short lengths the per-node Python overhead above dominates and the curve looks almost linear; attention's `T²` term only takes over further out. A single fitted exponent across the whole range would average two regimes into a number describing neither — which is why the slopes are local.

```
generating 60 tokens processed 1,830 positions
where a cached implementation processes 60  —  30.5x more work
```

Nothing here caches keys and values, so every step re-runs the entire prefix: `1 + 2 + ... + n` positions instead of `n`. **This is the single largest thing missing from the repo**, and it is an engineering absence rather than a gap in the mathematics.

## The honest account

It demonstrates that **the mathematics is right**: agreement to `1e-15` with an independent implementation, on derivatives nobody here copied.

It does not demonstrate that the code is fast, and was never going to. The gap would widen on a GPU, at a realistic model size, and against a KV cache this generation loop does not have.

The interesting part is which of those is the hard one. The calculus is what a person can derive, verify, and be sure of. The performance is what takes a compiler team — and buying it requires understanding none of what was derived here.

A good many claims in this repo were wrong before they were measured, and each is documented where it was made rather than quietly corrected:

| Day | What was wrong |
|-----|----------------|
| 7 | `backward()` silently cleared gradients, making `zero_grad()` dead code and contradicting Day 1's own README |
| 7 | Adam's bias correction described backwards — the uncorrected first step is 3.16× too *large* |
| 9 | A gradient probe read the wrong way round |
| 11 | A capacity demo a single head solved by memorising |
| 12 | A permutation-equivariance claim the measurement contradicted |
| 15 | Depth credited for what training does to the residual stream |
| 15 | A causality probe defeated by weight tying — 85 "leaks" that were nothing of the kind |
| 16 | A held-out split that was a copy of the training text |
| 16 | A perplexity measured past the window the model trained on |
| 17 | A backward pass timed at *minus three milliseconds* |

And one result that still stands against the repo: on 2,940 characters, **a count-based trigram scores a better perplexity than this model** and takes no training at all. The transformer's advantage is long-range structure, and that corpus has none. A repo reporting only the flattering half of that would be teaching the wrong lesson.

## Run it

```bash
# the comparison, the profile, and the writeup
python comparison.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 05_marking_the_homework/day17_pytorch_comparison

# tests — from inside this folder
python -m unittest
python test_comparison.py
```

PyTorch is the only third-party dependency in the repo and is needed only here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Without it the torch tests **skip** and the demo says so in plain words. CI installs it, because a green badge that stood for a check which never executed would be exactly the kind of thing this day exists to argue against.

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.
>
> ⚠ On Windows, NumPy and PyTorch may both link OpenMP and abort with `OMP: Error #15`. `KMP_DUPLICATE_LIB_OK=TRUE` works around it; the numbers above were verified identical to NumPy's under that flag.

## Where this leads

Nowhere — this is the last day. The repo is 17 days, 624 tests, and one working GPT built from `d/dx` upward.
