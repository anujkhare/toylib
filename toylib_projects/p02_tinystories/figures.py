"""Every figure in the post, rebuilt from checkpoints and logs.

    uv run python -m toylib_projects.p02_tinystories.figures --run local
    uv run python -m toylib_projects.p02_tinystories.figures --run local --only copying
"""

import argparse
import dataclasses
import pathlib
import textwrap

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from toylib import viz
from toylib.harness import inference
from toylib_projects.p02_tinystories import decoder_only_model
from toylib_projects.p02_tinystories import experiment
from toylib_projects.p02_tinystories import probes
from toylib_projects.p02_tinystories import tinystories
from toylib_projects.p02_tinystories import train

FIGURES_DIR = pathlib.Path(__file__).parent / "figures"

# From `--find-names 24`, kept as found. Written down before any model was measured on
# them; see probes.find_invented_names for the rules they pass.
INVENTED_NAMES = [
    " Graizzy", " Braibus", " Brouvle", " Stoolet", " Swindy", " Dralump", " Skinkab",
    " Spoundo", " Prorix", " Grutig", " Stezot", " Snuda", " Floutet", " Twousky",
    " Breepot", " Trilok", " Blourby", " Frogy", " Bloombo", " Treeret", " Spemy",
    " Twozy", " Starble", " Frilot",
]  # fmt: skip
NUM_CONTROLS = 6  # Other names each one is compared with; all of them would be slow.
COMMON_NAMES = [" Lily", " Tom", " Sue", " Ben", " Anna", " Max"]
NAME_TEMPLATES = [
    ("Once upon a time, there was a little girl named", ". She loved to play outside. One day,"),
    ("Once upon a time, there was a small dog named", ". He was very happy. Every morning,"),
    ("One day, a boy met a new friend called", ". They played all day. At night,"),
]
# Bands of training count for first_vs_repeat, from rare tokens to the commonest.
COUNT_EDGES = [0, 3_000, 30_000, 300_000, 10**9]
SAMPLE_PROMPT = "Once upon a time, there was a little girl named Lily. She"
ATTENTION_TEXT = (
    "Tom had a red ball. He gave the ball to his sister, Mia."
    ' Mia said, "Thank you, Tom!" She'
)
LENS_TEXT = "Lily and Ben went to the park. Lily saw a big dog and said hello to"
WORD_GROUPS = {
    "names": [" Lily", " Tom", " Sue", " Ben", " Anna", " Max", " Sara", " Tim"],
    "animals": [" dog", " cat", " bird", " fish", " bear", " rabbit", " frog", " mouse"],
    "colors": [" red", " blue", " green", " yellow", " pink", " purple", " orange", " black"],
    "feelings": [" happy", " sad", " angry", " scared", " proud", " tired", " excited", " kind"],
}


@dataclasses.dataclass
class Inspector:
    """One finished run: its configuration, its checkpoints and its tokenizer."""

    run: train.TinyStoriesRun
    run_dir: pathlib.Path
    data_dir: pathlib.Path = tinystories.DATA_DIR

    def __post_init__(self):
        self.checkpoint_dir = self.run_dir.resolve() / "checkpoints"
        self.tokenizer = tinystories.load_tokenizer(self.run.vocab_size, self.data_dir)
        self.steps = inference.checkpoint_steps(self.checkpoint_dir)
        self.val = np.memmap(
            tinystories.bin_path("val", self.run.vocab_size, self.data_dir),
            dtype=np.uint16,
            mode="r",
        )
        self._template = experiment.build_model(
            train.model_config(self.run), jax.random.key(0)
        )

    def model(self, step: int | None = None):
        return inference.load_model(self._template, self.checkpoint_dir, step)

    def tokens_seen(self, step: int) -> int:
        return (step + 1) * self.run.batch_size * self.run.seq_len

    def encode(self, text: str) -> np.ndarray:
        return np.array([tinystories.EOT_ID] + self.tokenizer([text])["input_ids"][0])

    def labels(self, tokens: np.ndarray) -> list[str]:
        strings = self.tokenizer.token_strings(tokens.tolist())
        strings = ["EOT" if s == "<|endoftext|>" else s for s in strings]
        # Escaped, since a pair of dollar signs would start math in tick labels.
        return [s.strip().replace("\n", "↵").replace("$", r"\$") or "·" for s in strings]

    def windows(self, count: int, length: int, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        offsets = rng.integers(0, len(self.val) - length, size=count)
        return np.stack([self.val[i : i + length] for i in offsets]).astype(np.int32)

    def stories(self, min_len: int, max_len: int, limit: int = 512) -> list[np.ndarray]:
        """Return validation stories within these lengths, each starting with its EOT."""
        head = np.asarray(self.val[:400_000])
        starts = np.flatnonzero(head == tinystories.EOT_ID)
        spans = [
            (a, b) for a, b in zip(starts, starts[1:]) if min_len <= b - a <= max_len
        ]
        return [head[a:b].astype(np.int32) for a, b in spans[:limit]]

    def story(self, index: int = 3, min_len: int = 70, max_len: int = 130) -> np.ndarray:
        """Return one validation story of a readable length."""
        return self.stories(min_len, max_len, limit=index + 1)[index]

    def train_counts(self) -> np.ndarray:
        """Return how often each token appears in the first 30M training tokens."""
        tokens = np.memmap(
            tinystories.bin_path("train", self.run.vocab_size, self.data_dir),
            dtype=np.uint16,
            mode="r",
        )
        return np.bincount(tokens[:30_000_000], minlength=self.run.vocab_size)


def fig_training(ins: Inspector) -> plt.Figure:
    """Loss over training, with the bytes it costs."""
    curves = viz.read_log(viz.find_log(ins.run_dir))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    steps, values = curves["train/loss"]
    ax.plot(steps, viz.ema(values, 0.9), color=viz.CATEGORICAL[0], label="train")
    steps, values = curves["val/loss"]
    ax.plot(steps, values, color=viz.CATEGORICAL[1], label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("nats per token")
    ax.set_title("Loss falls fastest in the first few hundred steps")
    ax.legend(loc="upper right")
    return fig


def fig_baselines(ins: Inspector) -> plt.Figure:
    """What counting achieves, against what the model achieves."""
    vocab = ins.run.vocab_size
    train_tokens = np.memmap(
        tinystories.bin_path("train", vocab, ins.data_dir), dtype=np.uint16, mode="r"
    )
    bits = probes.ngram_bits(
        np.asarray(train_tokens[:30_000_000]), np.asarray(ins.val), vocab
    )
    windows = ins.windows(128, ins.run.seq_len)
    model = ins.model()
    model_nats = np.mean([probes.token_nats(model, w[None]).mean() for w in windows])
    rows = [
        ("Every token equally likely", bits["uniform"]),
        ("Word frequency", bits["unigram"]),
        ("Previous token (bigram table)", bits["bigram"]),
        ("Previous two tokens (trigram table)", bits["trigram"]),
        (f"The model, after {ins.tokens_seen(ins.steps[-1]) / 1e6:.0f}M tokens", model_nats / np.log(2)),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    print(f"baselines: {rows}")
    labels, values = zip(*rows)
    y = np.arange(len(rows))[::-1]
    colors = [viz.INK_MUTED] * (len(rows) - 1) + [viz.CATEGORICAL[0]]
    ax.barh(y, values, color=colors, height=0.55)
    ax.set_yticks(y, labels=labels)
    for yi, value in zip(y, values):
        ax.text(value + 0.15, yi, f"{value:.2f}", va="center", color=viz.INK)
    ax.set_xlabel("bits per token")
    ax.grid(axis="y", visible=False)
    ax.set_title("Tables that see two tokens stop well short of the model")
    return fig


def fig_surprise(ins: Inspector) -> plt.Figure:
    """Where one story is expensive."""
    tokens = ins.story()
    model = ins.model()
    bits = probes.token_nats(model, tokens) / np.log(2)
    bands = probes.first_vs_repeat(
        model,
        ins.stories(100, 10**6),
        ins.train_counts(),
        COUNT_EDGES,
        length=min(200, ins.run.seq_len),
    )
    print(f"first_vs_repeat, bands {COUNT_EDGES}: {bands}")
    strings = ins.tokenizer.token_strings(tokens[1:].tolist())
    return viz.token_heatmap(
        strings,
        bits,
        vmax=12,
        line_width=78,
        colorbar_label="bits to predict this token",
        title="Cost of each token: a rare word gets cheaper once it has appeared",
    )


def fig_samples(ins: Inspector) -> plt.Figure:
    """One prompt, continued by the model at points in training."""
    wanted = [0, 10, 100, 1_000, ins.steps[-1]]
    wanted = sorted({min(ins.steps, key=lambda s: abs(s - w)) for w in wanted})
    prompt = ins.encode(SAMPLE_PROMPT)
    seq_len = ins.run.seq_len
    padded = jnp.zeros(seq_len, dtype=jnp.uint16).at[: len(prompt)].set(
        jnp.asarray(prompt, dtype=jnp.uint16)
    )
    rows = []
    for step in wanted:
        generated = decoder_only_model.jit_sample(
            ins.model(step),
            padded,
            len(prompt),
            jax.random.key(1),
            max_output_tokens=45,
            temperature=0.8,
            top_k=40,
        )
        text = ins.tokenizer.decode(generated.tolist()).replace("\n", " ")
        text = text.split("<|endoftext|>")[0].strip()
        rows.append((step, text))

    fig, ax = plt.subplots(figsize=(8, 0.95 * len(rows) + 0.5))
    ax.axis("off")
    ax.set_title(f'Continuing "{SAMPLE_PROMPT}"')
    for i, (step, text) in enumerate(rows):
        y = 1 - i / len(rows)
        ax.text(0, y, f"step {step:,}", va="top", fontweight="semibold", color=viz.INK_SECONDARY, transform=ax.transAxes)
        ax.text(0.14, y, textwrap.fill(text, 88), va="top", color=viz.INK, transform=ax.transAxes, fontsize=9, parse_math=False)
    return fig


def fig_attention(ins: Inspector) -> plt.Figure:
    """Each block's attention over one passage, averaged across heads."""
    tokens = ins.encode(ATTENTION_TEXT)
    _, acts = ins.model()(jnp.asarray(tokens), return_intermediates=True)
    labels = ins.labels(tokens)
    layers = ins.run.num_layers
    fig, axes = plt.subplots(1, layers, figsize=(4.4 * layers, 4.6))
    for i, ax in enumerate(np.atleast_1d(axes)):
        weights = np.asarray(acts[f"blocks.{i}.attn_weights"]).mean(axis=0)
        viz.heatmap(weights, row_labels=labels, col_labels=labels, vmin=0, vmax=1, ax=ax)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"Block {i + 1}")
    fig.suptitle("Averaged over heads, attention spreads thinly (rows read columns)", x=0.02, ha="left", fontweight="semibold")
    return fig


def fig_attention_distance(ins: Inspector) -> plt.Figure:
    """How far back each head looks."""
    max_distance = 64
    result = probes.attention_by_distance(
        ins.model(), ins.windows(64, ins.run.seq_len), max_distance
    )
    layers, heads = result.shape[:2]
    distances = np.arange(1, max_distance + 1)
    fig, axes = plt.subplots(1, layers, figsize=(2.9 * layers, 3), sharey=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        for head in range(heads):
            ax.plot(distances, result[i, head, 1:], color=viz.CATEGORICAL[0], alpha=0.45, linewidth=1.2)
        ax.plot(distances, result[i, :, 1:].mean(axis=0), color=viz.INK, linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
        ax.set_xlabel("tokens back")
        ax.set_title(f"Block {i + 1}")
    np.atleast_1d(axes)[0].set_ylabel("mean attention weight")
    fig.suptitle("The first block reads widely; later blocks stay close (black is the mean)", x=0.02, ha="left", fontweight="semibold", y=1.03)
    return fig


def fig_context(ins: Inspector) -> plt.Figure:
    """How much text the model actually uses."""
    lengths = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k < ins.run.seq_len - 1]
    lengths.append(ins.run.seq_len - 1)
    windows = ins.windows(512, ins.run.seq_len)
    model = ins.model()
    totals = [
        probes.context_curve(model, windows[i : i + 64], lengths)
        for i in range(0, 512, 64)
    ]
    values = [np.mean([t[k] for t in totals]) / np.log(2) for k in lengths]
    print(f"context: {dict(zip(lengths, np.round(values, 3).tolist()))}")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(lengths, values, color=viz.CATEGORICAL[0], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, labels=[str(k) for k in lengths])
    ax.set_xlabel("tokens of context the model may read")
    ax.set_ylabel("bits for the next token")
    ax.set_title("Reading further back helps up to about 64 tokens, then stops")
    return fig


def fig_lens(ins: Inspector) -> plt.Figure:
    """What the model would predict if it stopped at each depth."""
    tokens = ins.encode(LENS_TEXT)
    model = ins.model()
    lens = probes.lens_logits(model, tokens)
    depth_bits = probes.lens_nats(model, ins.windows(64, ins.run.seq_len)) / np.log(2)
    print(f"lens bits by depth: {np.round(depth_bits, 3).tolist()}")
    for depth in range(len(depth_bits)):
        margins = probes.name_probes(model, ins.tokenizer, INVENTED_NAMES, NAME_TEMPLATES, depth=depth, num_controls=NUM_CONTROLS)
        print(f"lens depth {depth}: finish {np.nanmean(margins['finish']):.2f}, recall {margins['recall'].mean():.2f}")
    shown = np.arange(len(tokens) - 12, len(tokens) - 1)
    rows = ["embedding"] + [f"block {i + 1}" for i in range(ins.run.num_layers)]
    probs = np.zeros((len(rows), len(shown)))
    tops = []
    for r, logits in enumerate(lens):
        p = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        probs[r] = p[shown, tokens[shown + 1]]
        tops.append(ins.tokenizer.token_strings(p[shown].argmax(-1).tolist()))
    columns = [f"{a}\n→ {b}" for a, b in zip(ins.labels(tokens[shown]), ins.labels(tokens[shown + 1]))]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    viz.heatmap(probs, row_labels=rows, col_labels=columns, vmin=0, vmax=1, ax=ax, colorbar_label="P(the true next token)")
    ax.tick_params(axis="x", labelsize=8)
    for r in range(len(rows)):
        for c in range(len(shown)):
            ink = "white" if probs[r, c] > 0.55 else viz.INK
            ax.text(c, r, tops[r][c].strip().replace("\n", "↵") or "·", ha="center", va="center", fontsize=8, color=ink, parse_math=False)
    ax.set_title("The model's top guess, if it stopped at each depth")
    return fig


def fig_copying(ins: Inspector) -> plt.Figure:
    """Does the model read what it was told? Measured at every checkpoint."""
    steps = [s for s in ins.steps if s > 0]
    series = {"finish": [], "recall": [], "common": [], "repeat": []}
    for step in steps:
        model = ins.model(step)
        invented = probes.name_probes(model, ins.tokenizer, INVENTED_NAMES, NAME_TEMPLATES, num_controls=NUM_CONTROLS)
        series["recall"].append(invented["recall"].mean())
        series["finish"].append(np.nanmean(invented["finish"]))
        common = probes.name_probes(model, ins.tokenizer, COMMON_NAMES, NAME_TEMPLATES)
        series["common"].append(common["recall"].mean())
        repeat = probes.repeat_test(model)
        series["repeat"].append(repeat["shuffled"] - repeat["repeated"])
    tokens = [ins.tokens_seen(s) for s in steps]
    names = {
        "finish": "finish an invented name",
        "recall": "bring an invented name back",
        "common": "bring a common name back",
        "repeat": "repeat a random string, in order",
    }
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhline(0, color=viz.INK_MUTED, linewidth=1)
    for i, (key, label) in enumerate(names.items()):
        ax.plot(tokens, series[key], color=viz.CATEGORICAL[i], marker="o", markersize=4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens seen")
    ax.set_ylabel("nats saved by having read it")
    ax.set_title("Reading a name changes what comes next; a random string doesn't")
    ax.legend(loc="upper left")
    for key, values in series.items():
        print(f"copying/{key}: {np.round(values, 3).tolist()} at steps {steps}")
    return fig


def fig_names(ins: Inspector) -> plt.Figure:
    """Margins say a name helped. Probabilities say whether the model would use it."""
    model = ins.model()
    invented = probes.name_probes(model, ins.tokenizer, INVENTED_NAMES, NAME_TEMPLATES, num_controls=NUM_CONTROLS)
    common = probes.name_probes(model, ins.tokenizer, COMMON_NAMES, NAME_TEMPLATES)
    panels = [
        ("Bring a common name back", COMMON_NAMES, common["recall_seen"], common["recall_other"]),
        ("Bring an invented name back", INVENTED_NAMES, invented["recall_seen"], invented["recall_other"]),
        ("Finish an invented name", INVENTED_NAMES, invented["finish_seen"], invented["finish_other"]),
    ]
    for label, result in (("common", common), ("invented", invented)):
        means = {k: round(float(result[k].mean()), 2) for k in ("recall_seen", "recall_other", "recall_none")}
        print(f"names/{label}: {means}")
    for label, names in (("common", COMMON_NAMES), ("invented", INVENTED_NAMES)):
        reuse = probes.name_reuse(model, ins.tokenizer, names, NAME_TEMPLATES)
        print(f"names/{label}: written back in {reuse:.1%} of samples")

    heights = [len(names) for _, names, _, _ in panels]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7, 1.4 + 0.17 * sum(heights)),
        sharex=True,
        gridspec_kw={"height_ratios": heights},
    )
    for ax, (title, names, seen, other) in zip(axes, panels):
        y = np.arange(len(names))[::-1]
        ax.hlines(y, np.exp(other), np.exp(seen), color=viz.INK_MUTED, linewidth=1.5)
        ax.scatter(np.exp(other), y, color=viz.CATEGORICAL[1], s=36, zorder=3, label="another name in the story")
        ax.scatter(np.exp(seen), y, color=viz.CATEGORICAL[0], s=36, zorder=3, label="this name in the story")
        ax.set_yticks(y, labels=[name.strip() for name in names], fontsize=8)
        ax.set_xscale("log")
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("probability of the name's next token")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Known names come back; most invented names stay unlikely", x=0.02, ha="left", fontweight="semibold")
    return fig


def fig_embeddings(ins: Inspector) -> plt.Figure:
    """Where groups of words sit, in the input embeddings and in the output layer."""
    model = ins.model()
    tokens, groups = [], []
    for group, words in WORD_GROUPS.items():
        for word in words:
            ids = ins.tokenizer([word])["input_ids"][0]
            if len(ids) == 1:
                tokens.append(ids[0])
                groups.append(group)
    # Each token owns a row of the embedding and a column of the output layer.
    tables = {
        "Input embeddings": np.asarray(model.embedding_layer.weights),
        "Output layer": np.asarray(model.output_layer.weights).T,
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (title, weights) in zip(axes, tables.items()):
        xy = probes.pca_2d(weights[tokens])
        for i, group in enumerate(WORD_GROUPS):
            at = [j for j, g in enumerate(groups) if g == group]
            ax.scatter(xy[at, 0], xy[at, 1], color=viz.CATEGORICAL[i], s=36, label=group, edgecolors=viz.SURFACE, linewidths=1)
            for j in at:
                ax.annotate(ins.labels(np.array([tokens[j]]))[0], xy[j], xytext=(5, 3), textcoords="offset points", fontsize=8, color=viz.INK_SECONDARY)
        ax.set_xlabel("first principal component")
        ax.set_ylabel("second principal component")
        ax.set_title(title)
        unit = weights / np.linalg.norm(weights, axis=-1, keepdims=True)
        print(f"embeddings/{title}: within-group cosine {_within_group_cosine(unit[tokens], groups):.3f}")
        for word in (" Lily", " dog", " red", " happy"):
            near = probes.nearest_tokens(weights, ins.tokenizer([word])["input_ids"][0][0], k=5)
            print(f"  {word} -> {ins.tokenizer.token_strings(near)}")
    axes[1].legend(loc="best")
    fig.suptitle("The output layer has sorted words into groups; the embeddings haven't", x=0.02, ha="left", fontweight="semibold")
    return fig


def _within_group_cosine(unit: np.ndarray, groups: list[str]) -> float:
    """Return the mean cosine over pairs of different tokens in the same group."""
    labels = np.array(groups)
    same = (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)
    return float((unit @ unit.T)[same].mean())


FIGURES = {
    "training": fig_training,
    "baselines": fig_baselines,
    "surprise": fig_surprise,
    "samples": fig_samples,
    "attention": fig_attention,
    "attention_distance": fig_attention_distance,
    "context": fig_context,
    "lens": fig_lens,
    "copying": fig_copying,
    "names": fig_names,
    "embeddings": fig_embeddings,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="local", choices=sorted(train.TINYSTORIES_RUNS))
    parser.add_argument("--run-dir", type=pathlib.Path, help="Default: outputs/<run>")
    parser.add_argument("--out", type=pathlib.Path, default=FIGURES_DIR)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    parser.add_argument("--find-names", type=int, help="Print this many invented names")
    args = parser.parse_args()

    viz.use_style()
    ins = Inspector(
        train.TINYSTORIES_RUNS[args.run],
        args.run_dir or train.OUTPUT_DIR / args.run,
    )
    if args.find_names:
        tokens = np.memmap(
            tinystories.bin_path("train", ins.run.vocab_size), dtype=np.uint16, mode="r"
        )
        print(probes.find_invented_names(ins.tokenizer, tokens, args.find_names))
        return
    for name in args.only or FIGURES:
        fig = FIGURES[name](ins)
        print(viz.save(fig, args.out / f"{name}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
