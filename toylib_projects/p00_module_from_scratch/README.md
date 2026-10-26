# Post 0: Building a tiny module system on JAX pytrees

Mostly drafted already, in two notebooks:

- [`00_jax_linear_regression.ipynb`](00_jax_linear_regression.ipynb): gradient descent in JAX on one variable.
- [`01_toylib_module.ipynb`](01_toylib_module.ipynb): from a hand-written `Linear` to the `Module` base class in `toylib/nn/module.py`.

**Headline question.** What is a pytree, and how did "every array is a
parameter" end up training our RoPE tables? The rule that makes `Module` short
(any array attribute is trainable) is also what made RoPE's sine and cosine
tables trainable in the GPT. Fixing it (review fix 1) and the related
list-of-arrays gap (fix 2) makes a good ending: a small, honest bug that teaches
how JAX sees a model.

**To finish:** notebook 01 stops at an earlier design, where each module writes
its own `__init__`. Extend it to the current one (dataclass fields, a separate
`init()`, nested modules initialized automatically), add the RoPE section with a
before/after check that the tables no longer move during training, and write
`POST.md`.
