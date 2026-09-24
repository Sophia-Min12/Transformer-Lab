# Transformer-Lab

![tests](https://github.com/Sophia-Min12/Transformer-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — a GPT built from scalars up, with every gradient checked against finite differences.**

> Autograd, softmax, attention, positional encoding, a character-level GPT that trains. Written in NumPy, derived by hand, and verified numerically at every step. No framework does the calculus for you until Level 5, and then only to mark the homework.

**Environment**: Python 3.10+ · NumPy from Level 1 · PyTorch in Level 5 only, as a reference to check against · `pytest` as the test runner.

---

## 🧭 The rule this repo is built on

Every derivative implemented here is checked against a **finite-difference approximation** of the same derivative:

```
df/dx  ≈  (f(x + h) − f(x − h)) / 2h
```

If the analytic gradient and the numerical one disagree beyond tolerance, the test fails. This is the single habit that makes backpropagation learnable rather than mysterious — you are never asked to trust a derivation, only to check it.

It is also how real frameworks are tested. `torch.autograd.gradcheck` does exactly this.

---

## 🗺️ Curriculum Roadmap

**Level 0 · Repo Setup**
- [x] **Day 0** — Repo scaffold, CI, and curriculum roadmap

**Level 1 · Autograd from Scratch**
- [x] **Day 1** — Scalar autograd: a `Value` that remembers how it was made
- [x] **Day 2** — More operations, and the finite-difference gradient check
- [x] **Day 3** — Tensor autograd: NumPy arrays and broadcasting
- [x] **Day 4** — Matrix multiplication and the chain rule in two dimensions

**Level 2 · The Pieces of a Network**
- [x] **Day 5** — Linear layers, initialization, and a `Module` base
- [x] **Day 6** — Softmax and cross-entropy, numerically stable
- [x] **Day 7** — SGD, Adam, and a training loop that learns something
- [x] **Day 8** — Embedding layers and one-hot without the one-hot

**Level 3 · Attention**
- [x] **Day 9** — Scaled dot-product attention
- [x] **Day 10** — Causal masking: why a language model may not look ahead
- [x] **Day 11** — Multi-head attention
- [x] **Day 12** — Positional encoding: attention has no sense of order

**Level 4 · A Working GPT**
- [ ] **Day 13** — LayerNorm and residual connections
- [ ] **Day 14** — The feed-forward network and the full transformer block
- [ ] **Day 15** — Character-level GPT: assemble and train
- [ ] **Day 16** — Sampling, temperature, and perplexity

**Level 5 · Marking the Homework**
- [ ] **Day 17** — PyTorch comparison, profiling, and the honest writeup

---

## 📐 Conventions

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: helpers from earlier days are copied forward with a `# reused from dayNN` comment, so no test ever needs a cross-folder import.
- Every day that introduces a derivative ships a **finite-difference test** for it.
- Every code day ships a runnable test; `pytest` from the repo root runs everything (this is what CI runs).
- Randomness is always seeded. A test that cannot be re-run to the same number is not a test.

## 🔗 Sibling labs

- [NLP-Lab](https://github.com/Sophia-Min12/NLP-Lab) — classical NLP. Its **Day 12 BPE tokenizer** is the input format this repo's GPT consumes, and its **Day 11 perplexity** is the metric Day 16 reuses.
- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — search & optimization. Gradient descent here is that repo's descent methods, applied to a loss surface with millions of parameters.

## License

[MIT](LICENSE) © Sophia Min
