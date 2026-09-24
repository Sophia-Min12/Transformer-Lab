# Day 8 · Embedding Layers, and One-Hot Without the One-Hot

> A network takes dense vectors. Language is a sequence of **discrete symbols**. The bridge is the simplest layer in the repo: a table with one learned vector per symbol.

## It is a one-hot matmul, and nobody does it that way

Both forms are implemented here so the comparison can be **run** rather than asserted. Vocabulary 50,000, dimension 768, a batch of 1,024 tokens:

```
lookup       3.97 ms
one-hot    540.90 ms   (136x slower)

the one-hot form does 39,321,600,000 multiply-adds, of which
786,432 are not by zero - 0.0020% of the work
```

Tests confirm the two produce **identical values and identical gradients** — the lookup is not an approximation, it is the same function with the zeros skipped.

## The backward pass is the interesting half

Forward is `table.data[indices]` — a memory read with no arithmetic at all. Backward has to **scatter** gradients back to the rows that were read, and that is where the one real mistake lives:

```python
grad[indices] += out.grad             # WRONG
np.add.at(grad, indices, out.grad)    # right
```

```
row 1 read 3 times -> np.add.at gives [3.0, 3.0]
plain fancy-index += gives            [1.0, 1.0]
```

NumPy's fancy-index assignment keeps only the **last** write when an index repeats. This is Day 1's `+=` rule in index form, and it fails the same way: silently, and in the direction that still trains.

It is worse here than it looks. In a language model the *common* words repeat constantly inside a single batch, so the wrong form under-counts exactly the rows with the most evidence — the model learns frequent words more slowly than rare ones, which is precisely backwards, and nothing reports it.

## Rows nobody reads never move

```
rows touched: 0, 1   gradient norms: [2.6939, 2.9886, 0.0, 0.0, 0.0]
```

An untouched row receives **exactly** zero gradient, so the optimizer leaves it at its initial value. A test checks the data is bit-identical after a step.

This is why a rare token's embedding is still essentially random after a full epoch — and NLP-Lab's Day 3 measured how much of a vocabulary that describes: around 60% of types occur exactly once. Most of an embedding table barely trains.

## Initialization is different here

`Embedding` uses a fixed `0.02` standard deviation rather than Day 5's Xavier. There is no fan-in to balance: each row is read independently and never participates in a dot product with its neighbours, so the variance argument that produced `√(2/(fan_in+fan_out))` simply does not apply. A small fixed scale is the convention, and GPT-2 uses exactly this number.

## Shapes

`(batch, time)` indices in, `(batch, time, dim)` out — which is the shape Day 9's attention expects, produced by a `reshape` after the flat gather rather than by a special case.

## Run it

```bash
# demo (equivalence, the cost, the scatter, unused rows)
python embeddings.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_network_pieces/day08_embeddings

# tests — from inside this folder
python -m unittest
python test_embeddings.py

# the docstring examples are runnable too
python -m doctest embeddings.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Level 2 closes here

A model can be built, initialized, given a loss, trained, and fed discrete symbols. Everything in it is a `Linear`, an activation, or a lookup — and every gradient in it has been checked against finite differences.

**Level 3** adds the one operation that is genuinely new: attention, where the weights applied to the input are **computed from the input itself** rather than learned once and fixed.
