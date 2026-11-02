"""Every figure renders from a tiny checkpointed run."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from toylib import viz
from toylib_projects.p02_tinystories import figures
from toylib_projects.p02_tinystories import tinystories
from toylib_projects.p02_tinystories import train
from toylib_projects.p02_tinystories.tokenizer import bpe

# Trained on the figures' word groups, which have to be single tokens.
WORDS = {word: 1 for words in figures.WORD_GROUPS.values() for word in words}
TOKENIZER = bpe.train(WORDS, 2_000)
RUN = train.TinyStoriesRun(
    num_layers=2,
    num_heads=2,
    qkv_dim=16,
    batch_size=2,
    max_steps=4,
    seq_len=128,
    vocab_size=len(TOKENIZER),
    warmup_steps=1,
    eval_interval_steps=2,
    num_eval_steps=1,
    save_interval_steps=100,
)


@pytest.fixture(scope="module")
def inspector(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    data_dir = root / "data"
    data_dir.mkdir()
    TOKENIZER.save(tinystories.tokenizer_path(RUN.vocab_size, data_dir))
    tokens = np.random.default_rng(0).integers(2, 64, 8_000, dtype=np.uint16)
    tokens[::100] = tinystories.EOT_ID  # Stories of 100 tokens.
    for split in ("train", "val"):
        tokens.tofile(tinystories.bin_path(split, RUN.vocab_size, data_dir))

    exp = train.create_tinystories_experiment(
        RUN, run_id="test", data_dir=data_dir, output_dir=root
    )
    exp.outer_loop()
    exp.cleanup()
    viz.use_style()
    return figures.Inspector(RUN, root / "test", data_dir)


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_figure_renders(inspector, name, tmp_path):
    fig = figures.FIGURES[name](inspector)
    saved = viz.save(fig, tmp_path / f"{name}.png")
    plt.close(fig)
    assert saved[0].stat().st_size > 0


def test_story_starts_with_eot(inspector):
    assert inspector.story(index=0)[0] == tinystories.EOT_ID
