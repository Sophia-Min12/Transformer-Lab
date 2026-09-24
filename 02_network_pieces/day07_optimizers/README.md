# Day 7 · SGD, Adam, and a Training Loop That Learns Something

> Everything needed to train has existed since Day 6. This day adds the four lines that use it, and produces the first thing in this repo that actually learns.

```python
loss = loss_fn(model(x), targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The optimizer never sees the model — it is handed a list of tensors by `Module.parameters()` and updates them in place. That is why Day 5 spent its effort on making the list complete.

## It learns XOR

```
XOR, 82 parameters, 400 points
loss 0.7327 -> 0.0001   (chance is log 2 = 0.6931)
accuracy 100.0%
```

XOR is the smallest problem a linear model **provably** cannot solve — no straight line separates the four corners. So the companion result matters as much:

```
a linear model:  loss 0.6925 -> 0.6868   accuracy 55.2%
```

It sits at chance, which is the *correct* answer for a line on XOR rather than a bug in the loop. A test asserts both, which means the loop is verified to learn something a weaker model cannot.

## A defect this day exposed in Day 1

Writing the `zero_grad` demonstration surfaced a real bug in the autograd core: **`backward()` was clearing every gradient in the graph before propagating.**

That made three things wrong at once. `optimizer.zero_grad()` was dead code. Day 1's README claim that "gradients accumulate across calls" was false. And it diverged from PyTorch, which Day 17 compares against — PyTorch accumulates, which is exactly why `zero_grad()` exists there.

`backward()` now seeds the output and adds into every `.grad` without clearing first. All 255 existing tests still passed, and Day 1 gained two tests pinning the corrected behaviour.

## What forgetting `zero_grad` actually does

```
with zero_grad     final loss 0.6797  accuracy 55.2%
without            final loss 0.0001  accuracy 100.0%
```

It does not crash. At step *t* the update applies the **sum of every gradient so far**, which is an effective learning rate that grows with the step count.

Here that happened to produce a *better* result — SGD at `lr=0.05` was simply too slow for 60 steps, and the accidental amplification helped. **That is exactly why the bug is dangerous:** it can present as an improvement. A test pins only that the two runs differ, and a second test shows the accumulation directly — two `backward()` calls give exactly twice the gradient.

## Adam's bias correction is described backwards everywhere

```
corrected    first step = 1.0000
uncorrected  first step = 3.1623
```

Both running averages start at zero, which biases them toward zero early. The usual telling implies the first steps are therefore **too small**. They are not:

```
(1 - β₁) / √(1 - β₂)  =  0.1 / 0.0316  =  3.16
```

The uncorrected first step is **3.16× too large**. The bias divides out of the ratio, and because `m` is biased by `(1−β₁)` while `√v` is biased by `√(1−β₂)`, the two do not cancel — they compound in the wrong direction.

With correction the step is exactly `lr` at every `t`, which a test checks over 50 steps.

## Where Adam earns its keep

One feature scaled 100× larger than the other:

```
optimizer               final loss   accuracy
SGD lr=1e-4                 0.6115      74.7%
SGD lr=1e-4 + mom .9        0.5503      74.3%
Adam lr=0.05                0.0134      99.3%
```

Adam divides each parameter's step by its own running gradient scale, so a feature 100× larger does not force a 100× smaller learning rate for everything else. Momentum helps SGD a little and does not fix the underlying conditioning.

The invariance is **almost** exact: `eps` sits in the denominator unscaled, so it costs a small gradient about `1e-6` of its step. A test identifies `eps` as the cause by shrinking it and watching the agreement tighten by six orders of magnitude.

## Learning rate: too large stops being gradient descent

```
       lr     final   worst seen  steps worse
     0.01    0.6865       0.7327            0
     0.10    0.3760       0.7327            0
     1.00    0.0031       0.7327            0
    10.00    0.0000      20.2114            5
   100.00    0.0000     235.4880            6
```

I expected this table to show divergence to `inf`/`nan`. **It does not, and that is the finding.** `tanh` bounds every activation, so an absurd step cannot blow the network up — it only teleports the parameters somewhere arbitrary.

What it *does* show is the loss climbing to **235** on the way, and going uphill on several steps. That is no longer descent in any meaningful sense, and the fact that it lands somewhere low anyway is luck. Four tests pin the shape: a sane rate never goes uphill, a huge one does, it overshoots more than 10× its starting loss, and nothing ever reaches `inf`.

## Run it

```bash
# demo (XOR, bias correction, scaling, learning rate, zero_grad)
python optimizers.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_network_pieces/day07_optimizers

# tests — from inside this folder
python -m unittest
python test_optimizers.py

# the docstring examples are runnable too
python -m doctest optimizers.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The network takes dense vectors. Language is a sequence of **discrete symbols**, and turning one into the other with a one-hot matrix multiply wastes almost all of the work. **Day 8** builds the embedding layer, whose forward pass is an array index and whose backward pass is the first genuinely interesting scatter in this repo.
