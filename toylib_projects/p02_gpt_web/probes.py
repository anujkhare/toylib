"""Measurements on a trained model. `figures.py` draws them; this file only computes.

Each probe takes a model (any `DecoderOnlyTransformer`) and returns plain arrays,
so a checkpoint sweep is a loop over `inference.load_model`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from toylib_projects.p02_gpt_web import decoder_only_model

_forward = jax.jit(lambda model, tokens: model(tokens))
_forward_acts = jax.jit(lambda model, tokens: model(tokens, return_intermediates=True))
_lens = jax.jit(lambda model, hidden: model.logits(hidden))


def _nats(logits: jax.Array, targets: np.ndarray) -> np.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, jnp.asarray(targets)[..., None], axis=-1)
    return -np.asarray(picked.squeeze(-1))


def token_nats(model, tokens: np.ndarray) -> np.ndarray:
    """Return the cost in nats of each token after the first, given what came before."""
    return _nats(_forward(model, jnp.asarray(tokens[..., :-1])), tokens[..., 1:])


def ngram_bits(
    train: np.ndarray, val: np.ndarray, vocab_size: int
) -> dict[str, float]:
    """Return bits per token on `val` of lookup tables built from `train`.

    Each order backs off to shorter contexts by fixed-weight interpolation, so
    unseen contexts don't cost infinite bits. No model beats these by luck: they are
    what plain counting achieves.
    """
    train, val = train.astype(np.int64), val.astype(np.int64)

    def table(tokens: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
        keys = tokens[: len(tokens) - order + 1].copy()
        for i in range(1, order):
            keys = keys * vocab_size + tokens[i : len(tokens) - order + 1 + i]
        return np.unique(keys, return_counts=True)

    def lookup(keys: np.ndarray, counts: np.ndarray, query: np.ndarray) -> np.ndarray:
        at = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
        return np.where(keys[at] == query, counts[at], 0).astype(np.float64)

    def conditional(order: int) -> np.ndarray:
        """Return P(val[t] | previous order-1 tokens) for t >= 2."""
        full = val[2:]
        context = np.zeros_like(full)
        for i in range(1, order):
            full = full + val[2 - i : len(val) - i] * vocab_size**i
            if i > 1:
                context = context + val[2 - i : len(val) - i] * vocab_size ** (i - 1)
            else:
                context = val[1 : len(val) - 1]
        numerator = lookup(*table(train, order), full)
        if order == 1:
            return numerator / len(train)
        denominator = lookup(*table(train, order - 1), context)
        return numerator / np.maximum(denominator, 1)

    uniform = np.full(len(val) - 2, 1.0 / vocab_size)
    probs = {order: conditional(order) for order in (1, 2, 3)}
    mix = {
        "unigram": 0.999 * probs[1] + 0.001 * uniform,
        "bigram": 0.9 * probs[2] + 0.099 * probs[1] + 0.001 * uniform,
        "trigram": 0.7 * probs[3] + 0.25 * probs[2] + 0.049 * probs[1] + 0.001 * uniform,
    }
    bits = {"uniform": float(np.log2(vocab_size))}
    bits.update({k: float(-np.log2(p).mean()) for k, p in mix.items()})
    return bits


def attention_by_distance(model, tokens: np.ndarray, max_distance: int) -> np.ndarray:
    """Return the mean attention `distance` tokens back.

    Result has shape [num_layers, num_heads, max_distance + 1]; column 0 is a token
    attending to itself.
    """
    _, acts = _forward_acts(model, jnp.asarray(tokens))
    rows = []
    for layer in range(model.config.num_layers):
        weights = np.asarray(acts[f"blocks.{layer}.attn_weights"])  # [B, H, T, T]
        rows.append(
            [
                np.diagonal(weights, offset=-d, axis1=-2, axis2=-1).mean(axis=(0, 2))
                for d in range(max_distance + 1)
            ]
        )
    return np.array(rows).transpose(0, 2, 1)


def lens_logits(model, tokens: np.ndarray) -> list[np.ndarray]:
    """Read predictions off every depth, by sending it through the output layer.

    Entry 0 uses the embeddings alone, and entry k the state after block k, so the
    last entry is the model's real output. Earlier entries weren't trained to
    predict, but the residual stream makes them readable (the "logit lens").
    """
    _, acts = _forward_acts(model, jnp.asarray(tokens))
    names = ["embedding"]
    names += [f"blocks.{i}.hidden" for i in range(model.config.num_layers)]
    return [np.asarray(_lens(model, acts[name])) for name in names]


def lens_nats(model, tokens: np.ndarray) -> np.ndarray:
    """Return the mean loss if the model stopped at each depth."""
    targets = tokens[..., 1:]
    return np.array(
        [_nats(l[..., :-1, :], targets).mean() for l in lens_logits(model, tokens)]
    )


def context_curve(
    model, windows: np.ndarray, lengths: list[int]
) -> dict[int, float]:
    """Return nats on each window's last token, seeing only the `k` tokens before it.

    RoPE is relative, so a cut-off context needs no other change.
    """
    target = windows[:, -1]
    result = {}
    for k in lengths:
        inputs = jnp.asarray(windows[:, -1 - k : -1])
        result[k] = float(_nats(_forward(model, inputs)[:, -1], target).mean())
    return result


def repeat_test(
    model,
    length: int = 40,
    num_sequences: int = 256,
    vocab_range: tuple[int, int] = (257, 2_000),
    eot: int = 0,
) -> dict[str, float]:
    """Compare a random string's second copy with the same tokens shuffled.

    This is the classic induction-head test: the tokens are random, so the only way
    to beat the shuffle is to have read the first copy in order. An unrelated string
    would be a weaker control, since a token that has appeared gets cheaper however
    it's placed. The default range is learned tokens, as lone bytes are often not text.
    """
    rng = np.random.default_rng(0)
    first = rng.integers(*vocab_range, size=(num_sequences, length))
    start = np.full((num_sequences, 1), eot)
    repeated = token_nats(model, np.concatenate([start, first, first], axis=1))
    shuffled = rng.permuted(first, axis=1)
    shuffled = token_nats(model, np.concatenate([start, first, shuffled], axis=1))
    return {
        "repeated": float(repeated[:, length + 1 :].mean()),
        "shuffled": float(shuffled[:, length + 1 :].mean()),
    }


def name_probes(
    model,
    tokenizer,
    names: list[str],
    templates: list[tuple[str, str]],
    depth: int | None = None,
    num_controls: int | None = None,
) -> dict[str, np.ndarray]:
    """Measure whether a name in the context changes what comes next.

    Each template is `(before, after)`, and a name goes between them. Every name is
    tested against every other as a control, so a name's own frequency cancels out.

    * `recall`: at the slot after `after`, log P(first token of the name) when that
      name was in the context, minus the same when another name was.
    * `finish`: the name has started again, so log P(its last token) when it was
      in the context, minus the same when another name was. This is [A][B]...[A]->[B].

    A margin says how much the name helped, not whether the model would now say it,
    so the log probabilities behind each are returned too, as `*_seen` and `*_other`,
    along with `recall_none`: the same slot with no name in the context at all.
    `depth` reads each prediction through the logit lens instead of from the output.
    `num_controls` compares each name with only the next few, to save time.

    Returns arrays with one entry per name (averaged over templates and controls);
    the `finish` arrays are NaN for one-token names.
    """
    encode = lambda text: tokenizer([text])["input_ids"][0]  # noqa: E731
    ids = {name: encode(name) for name in names}

    def log_prob(context: list[int], target: int) -> float:
        if depth is None:
            logits = _forward(model, jnp.asarray(context))[-1]
        else:
            logits = lens_logits(model, np.array(context))[depth][-1]
        return float(jax.nn.log_softmax(logits)[target])

    keys = ("recall_seen", "recall_other", "recall_none", "finish_seen", "finish_other")
    rows = {key: [] for key in keys}
    for position, name in enumerate(names):
        row = {key: [] for key in keys}
        others = names[position + 1 :] + names[:position]
        for before, after in templates:
            head, tail = [0] + encode(before), encode(after)
            row["recall_none"].append(log_prob(head + tail, ids[name][0]))
            for other in others[:num_controls]:
                seen, unseen = head + ids[name] + tail, head + ids[other] + tail
                target = ids[name][0]
                row["recall_seen"].append(log_prob(seen, target))
                row["recall_other"].append(log_prob(unseen, target))
                if len(ids[name]) > 1:
                    start, last = ids[name][:-1], ids[name][-1]
                    row["finish_seen"].append(log_prob(seen + start, last))
                    row["finish_other"].append(log_prob(unseen + start, last))
        for key in keys:
            rows[key].append(np.mean(row[key]) if row[key] else np.nan)
    result = {key: np.array(values) for key, values in rows.items()}
    result["recall"] = result["recall_seen"] - result["recall_other"]
    result["finish"] = result["finish_seen"] - result["finish_other"]
    return result


def name_reuse(
    model,
    tokenizer,
    names: list[str],
    templates: list[tuple[str, str]],
    num_samples: int = 4,
    max_output_tokens: int = 60,
) -> float:
    """Return the fraction of sampled continuations that write the name in their prompt."""
    seq_len = model.config.seq_len
    hits, total = 0, 0
    for n, name in enumerate(names):
        for t, (before, after) in enumerate(templates):
            prompt = tokenizer([before + name + after])["input_ids"][0]
            prompt = jnp.asarray([0] + prompt, dtype=jnp.uint16)
            padded = jnp.zeros(seq_len, dtype=jnp.uint16).at[: len(prompt)].set(prompt)
            for seed in range(num_samples):
                generated = decoder_only_model.jit_sample(
                    model,
                    padded,
                    len(prompt),
                    jax.random.key(100 * n + 10 * t + seed),
                    max_output_tokens=max_output_tokens,
                    temperature=0.8,
                    top_k=40,
                )
                text = tokenizer.decode(generated.tolist()).split("<|endoftext|>")[0]
                hits += name.strip() in text
                total += 1
    return hits / total


def first_vs_repeat(
    model, documents: list[np.ndarray], counts: np.ndarray, edges: list[int], length: int = 200
) -> list[dict[str, float]]:
    """Compare a token's cost at its first appearance in a document and at its repeats.

    Common tokens are both repeated more and cheaper anyway, so the comparison is made
    within bands of `counts`, how often each token appears in training: one entry per
    pair of consecutive `edges`, with mean bits under `first` and `repeat`.
    """
    tokens, repeats, nats = [], [], []
    documents = [document[:length] for document in documents]
    for i in range(0, len(documents), 64):
        chunk = documents[i : i + 64]
        # Padded at the end, where causality keeps it from touching the costs we read.
        batch = np.stack([np.pad(doc, (0, length - len(doc))) for doc in chunk])
        costs = token_nats(model, batch)
        for doc, cost in zip(chunk, costs):
            seen = {int(doc[0])}
            for position, token in enumerate(doc[1:].tolist()):
                tokens.append(token)
                repeats.append(token in seen)
                nats.append(cost[position])
                seen.add(token)
    tokens, repeats, bits = np.array(tokens), np.array(repeats), np.array(nats) / np.log(2)
    bands = []
    for low, high in zip(edges, edges[1:]):
        band = (counts[tokens] >= low) & (counts[tokens] < high)
        first, repeat = bits[band & ~repeats], bits[band & repeats]
        bands.append(
            {
                "first": float(first.mean()) if len(first) else np.nan,
                "repeat": float(repeat.mean()) if len(repeat) else np.nan,
            }
        )
    return bands


def find_invented_names(
    tokenizer, train_tokens: np.ndarray, count: int, seed: int = 0
) -> list[str]:
    """Generate names that can test copying, keeping every one that passes.

    A name qualifies if it has two to four tokens, none of its token pairs ever occurs
    in training (so finishing it can't come from memory), its first token has at least
    two letters (a lone ` T` starts too many real names) and it shares no token with
    the names before it (since each serves as a control for the others).
    """
    vocab_size = len(tokenizer)
    pairs_seen = np.zeros(vocab_size * vocab_size, dtype=bool)
    for start in range(0, len(train_tokens), 50_000_000):
        chunk = np.asarray(train_tokens[start : start + 50_000_001]).astype(np.int64)
        pairs_seen[chunk[:-1] * vocab_size + chunk[1:]] = True

    rng = np.random.default_rng(seed)
    onsets = ["Bl", "Br", "Dr", "Fl", "Fr", "Gl", "Gr", "Kl", "Kr", "Pl", "Pr", "Sk", "Sn",
              "Sp", "St", "Sw", "Tr", "Tw", "Vr", "Wr", "Zl", "Zr"]  # fmt: skip
    vowels = ["a", "e", "i", "o", "u", "oo", "ee", "ai", "ou"]
    middles = ["b", "d", "g", "k", "l", "m", "n", "p", "r", "t", "v", "z", "zz", "mb",
               "nd", "rb", "lk", "sk", "nk"]  # fmt: skip
    endings = ["y", "o", "a", "et", "le", "ix", "us", "ump", "ab", "ok", "ot", "ig"]

    names, used = [], set()
    for _ in range(50_000):
        name = " " + "".join(
            str(rng.choice(part)) for part in (onsets, vowels, middles, endings)
        )
        ids = tokenizer([name])["input_ids"][0]
        new_pairs = all(not pairs_seen[a * vocab_size + b] for a, b in zip(ids, ids[1:]))
        first_is_long = len(tokenizer.decode(ids[:1]).strip()) >= 2
        if 2 <= len(ids) <= 4 and new_pairs and first_is_long and not used & set(ids):
            names.append(name)
            used |= set(ids)
            if len(names) == count:
                break
    return names


def nearest_tokens(weights: np.ndarray, token: int, k: int = 8) -> list[int]:
    """Return the `k` tokens with the highest cosine similarity to `token`."""
    unit = weights / np.linalg.norm(weights, axis=-1, keepdims=True)
    order = np.argsort(-(unit @ unit[token]))
    return [int(i) for i in order if i != token][:k]


def pca_2d(points: np.ndarray) -> np.ndarray:
    """Project rows onto their two largest principal components."""
    centered = points - points.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T
