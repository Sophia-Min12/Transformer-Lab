# Day 15 · Character-Level GPT: Assemble and Train

> Nothing new is built today. Every piece — the autograd, softmax, Adam, the embedding table, attention, the mask, multi-head, positional encoding, LayerNorm, the block — was written and gradient-checked on an earlier day. Day 15 wires them together and turns the crank.

```
model: dim 32, 2 blocks, 4 heads, 25,856 parameters

before training: loss 3.0033   log(20) = 2.9957
after 300 steps: loss 0.9354
perplexity 19.8 -> 2.5          (uniform over 20 characters is 20)
11.3 s
```

```
'the cat ' -> 'the cat the log the log the cat toheatoheatoheat'
'the mat ' -> 'the mat the log the log the cat toheatoheatoheat'
```

Not good text. But `the cat`, `the log` and the spacing are the corpus's, learned from 2,940 characters by 25,856 parameters in eleven seconds of NumPy — and the degeneration into `toheatoheat` is exactly what greedy decoding does, which is Day 16's subject.

## The starting loss is the cheapest bug you will ever catch

A fresh classifier over `V` classes knows nothing, so it should spread its probability evenly and score `log V`.

```
loss 3.0033 ; log(20) = 2.9957
```

If this number comes out at 6 instead of 3, the model is broken before a single step of training — and finding that out in one forward pass beats watching a loss curve fail to move for an hour.

## The offset

```
logits from positions 0..n-2, targets from tokens 1..n-1

window 'the ca'
  inputs  'the c'
  targets 'he ca'
```

Position `t` predicts token `t+1`. Get it wrong by one and the model is asked to predict the token it was just handed — a problem it can solve, which is what makes the bug dangerous: **the loss goes down**, and nothing raises an error. A test trains both versions for sixty steps and asserts the wrong one ends *lower*.

## Weight tying

```
tied   25,856 parameters
untied 26,496 parameters  (+640)
```

The embedding table and the output projection have the same shape transposed, and both answer the same question — how close is this vector to the vector for token *k*. Sharing them saves `vocab × dim`, a tenth of a real model's parameters. Day 5's parameter walk deduplicates by identity, so the shared tensor receives one optimizer update per step, not two; a test pins that.

## Two things I had wrong

**The final LayerNorm — depth was not the reason.**

```
|logit| with it 12.65 ; without 265.90 ; ratio 21.0x   (trained model)

a fresh model, by depth:
   1 block   1.68x
   2 blocks  2.86x
  16 blocks  4.17x
```

Day 13 measured the residual stream growing with depth, so I wrote that depth is why the final norm is needed. My own test then failed at 2.86× against an asserted 5×, and measuring properly says depth buys a factor of four across *sixteen* blocks while **training buys twenty across two**. It is learned weights that grow the stream. Both halves are now tests.

**The causality probe — Day 10's gradient trick does not survive weight tying.**

The first version counted embedding rows that received gradient and reported **85 leaks** in a model that is provably causal. The reason is the tying above: the embedding table is *also* the output projection, so every row gets gradient from the vocabulary softmax at every position, whether or not it was ever an input. The probe was measuring the tie, not the mask.

```
change the token at position 5:
  logits at positions 0-4 move by 0.00e+00
  logits at positions 5-7 move by 3.14e-03
```

Perturb an input and watch the earlier outputs. Exactly zero, and it survives tying. A test also pins the broken probe, so nobody reintroduces it.

## Run it

```bash
# demo (starting loss, training, generation, tying, the two corrections)
python gpt.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_working_gpt/day15_character_gpt

# tests — from inside this folder
python -m unittest
python test_gpt.py

# the docstring examples are runnable too
python -m doctest gpt.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The model trains, and greedy decoding makes it repeat itself. **Day 16** adds sampling and temperature, and measures perplexity properly — the metric [NLP-Lab's Day 11](https://github.com/Sophia-Min12/NLP-Lab) built for n-gram models, now pointed at a transformer.
