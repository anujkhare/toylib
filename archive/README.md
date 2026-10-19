# Archive

Earlier experiments, kept for reference. Nothing here is imported, tested or
maintained, and most of it no longer runs against the current `toylib`.

| File | What it was | Why it's here |
| --- | --- | --- |
| `notebooks/02. NN discriminative.ipynb` | Linear regression with an early `Linear` | Written for an older `Module` API (positional constructor arguments) |
| `notebooks/03. Text Classification - BOW Glove + MLP.ipynb` | IMDB sentiment with GloVe and an MLP | Older `Module` API (custom `__init__`) |
| `notebooks/04. Text Classification - Attention + Glove.ipynb` | IMDB sentiment with self-attention | Older `Module` API (`__post_init__`) |
| `notebooks/06. ConvNet - MNIST.ipynb` | MNIST data loading and a plotting helper; no model | Superseded by `toylib_posts/p01_cnn_mnist` |
| `notebooks/07. GAN - MNIST.ipynb` | MNIST data loading; no model | Starting point for a future GAN post |
| `notebooks/stable_diffusion.ipynb` | Exploring Stable Diffusion | Not built on toylib |
| `imdb.py` | IMDB + GloVe loader for notebooks 03 and 04 | Was `toylib/data/imdb.py`; pulled in torch and nltk for these notebooks only |

The old `toylib/trainer.py` was deleted rather than archived: `toylib.harness`
replaces it.
