# Day 11 · Multi-Head Attention

> A single head computes **one** distribution over positions per query and applies it to every output coordinate alike. It can express one relationship — "the adjective before me" *or* "the subject of this clause" — never both at once.

Splitting the width into `h` heads gives `h` independent distributions, each governing its own slice of the output.

## It is free

```
1 head(s): 256 parameters, weights (1, 5, 5)
2 head(s): 256 parameters, weights (2, 5, 5)
4 head(s): 256 parameters, weights (4, 5, 5)
8 head(s): 256 parameters, weights (8, 5, 5)
```

`h` heads of width `d/h` perform the same multiplications as one head of width `d`, the parameter count is **identical** (four `d×d` projections however the width is sliced), and Day 4's matmul already batches over leading axes so every head runs in one call. A test asserts the counts are equal across `h ∈ {1,2,4,8}`.

The only new operation is `permute`, whose backward pass — like `reshape` and `transpose` — contains **no arithmetic at all**. A permutation moves numbers; its derivative moves them back.

## Getting the demonstration to say anything took three attempts

This is the useful part of the day, and all three versions are preserved in the docstring rather than quietly replaced.

**Attempt 1 — one fixed sequence.** A single head reached a loss of `0.00000`, *beating* two heads. The structural argument was fine; the experiment was worthless. Attention weights are computed **from `x`**, so with one example and ten positions the model simply memorized the answer. Capacity claims need data the model has not seen.

**Attempt 2 — a positional rule.** I set `target[t, 0] = x[t-1, 0]`, "the previous position". Nothing could solve it, because **attention is permutation-equivariant and has no idea what position it is at** — that is Day 12's entire subject. I had written a task that needs a component the model does not have yet.

**Attempt 3 — content-addressed lookups, held-out split, enough data.**

```
target[:, 0] = x[argmax(x[:, 2]), 0]     position flagged by feature 2
target[:, 1] = x[argmax(x[:, 3]), 1]     position flagged by feature 3
```

Both lookups are content-based, so no notion of position is required. One head cannot read coordinate 0 from one flagged position while reading coordinate 1 from another.

```
  heads       train    held-out   (200 training sequences)
      1     0.75455     1.15184
      2     0.70247     1.04126
      4     0.53272     0.97293
```

Both columns fall. At 36 training sequences held-out went the *wrong* way — more heads overfit — and only more data separated capacity from memorization.

And the learned heads do the expected thing:

```
the two flagged positions are (5, 2)
head 1 attends most to position 5 (weight 0.98)
```

## The output projection is not decoration

`MultiHeadAttention` ends with a fourth `d×d` projection. Without it each head's slice would pass to the next layer untouched by the others, and the heads could never **combine** — only sit side by side. It is what turns `h` separate attentions into one layer.

## Run it

```bash
# demo (splitting, cost, the capacity table, what the heads learned)
python multi_head.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_attention/day11_multi_head

# tests — from inside this folder
python -m unittest
python test_multi_head.py

# the docstring examples are runnable too
python -m doctest multi_head.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Attempt 2 failed for a reason worth its own day. **Attention is permutation-equivariant**: shuffle the input positions and the outputs shuffle with them, unchanged. A model built from attention alone cannot tell *"dog bites man"* from *"man bites dog"*, which disqualifies it as a language model. **Day 12** measures that invariance and then breaks it.
