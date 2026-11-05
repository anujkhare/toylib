"""Fast sampling with a key-value cache, for chatting with a trained model.

`decoder_only_model.sample` recomputes the whole context for every token, which is fine
on a GPU but far too slow for a 477M model on a laptop. Here each new token runs through
the model once, attending to the keys and values cached from earlier positions.

`forward` runs the model's own layers in the same order as `DecoderOnlyTransformer.hidden`,
so it needs no changes to the model. `generate_test.py` checks that its logits match
`DecoderOnlyTransformer.__call__`; a change to the model's forward pass must be mirrored
here.

    gen = Generator(model)
    for token in gen.stream(prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        ...
"""

import dataclasses
import functools
import typing

import einops
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from toylib.nn import layers
from toylib_projects.p02_gpt_web import decoder_only_model


class Cache(typing.NamedTuple):
    keys: jax.Array  # [num_layers, num_heads, seq_len, head_dim]
    values: jax.Array  # [num_layers, num_heads, seq_len, head_dim]
    # The last token's normalized embedding, before smearing: what the next token mixes in.
    last_embedding: jax.Array  # [qkv_dim]


def empty_cache(
    model: decoder_only_model.DecoderOnlyTransformer,
    length: int | None = None,
    batch: tuple[int, ...] = (),
) -> Cache:
    """A cache for `length` positions (default: the whole context), with leading `batch` dims.

    A shorter cache is exact: no query can see a key beyond the positions it holds.
    """
    c = model.config
    shape = (*batch, c.num_layers, c.num_heads, length or c.seq_len, c.qkv_dim // c.num_heads)
    return Cache(
        jnp.zeros(shape, c.dtype), jnp.zeros(shape, c.dtype), jnp.zeros((*batch, c.qkv_dim), c.dtype)
    )


def _rope(rope, x: jax.Array, pos: jax.Array) -> jax.Array:
    """`RotaryPositionalEmbedding.__call__` from a traced start position."""
    t, d = x.shape[-2:]
    cos = jax.lax.dynamic_slice_in_dim(rope.cos, pos, t).astype(rope.dtype)
    sin = jax.lax.dynamic_slice_in_dim(rope.sin, pos, t).astype(rope.dtype)
    x1, x2 = x[..., : d // 2].astype(rope.dtype), x[..., d // 2 :].astype(rope.dtype)
    return jnp.concatenate([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], axis=-1)


def _attention(attn, x, value_embedding, layer, pos, cache: Cache):
    """`CausalSelfAttention.__call__` for positions pos..pos+t, reading and writing the cache."""
    mha, t = attn.mha, x.shape[-2]
    split = "t (h d) -> h t d"
    q, k, v = (
        einops.rearrange(proj(x), split, h=attn.num_heads)
        for proj in (mha.q_projection, mha.k_projection, mha.v_projection)
    )
    if value_embedding is not None:
        channels = attn.value_gate.shape[0]
        gate = 3 * jax.nn.sigmoid(x[..., :channels] @ attn.value_gate.astype(x.dtype))
        ve = einops.rearrange(value_embedding, split, h=attn.num_heads)
        v = v + einops.rearrange(gate, "t h -> h t 1") * ve
    q, k = _rope(attn.rope, q, pos), _rope(attn.rope, k, pos)
    q, k = layers.rms_norm(q) * attn.qk_scale, layers.rms_norm(k) * attn.qk_scale

    keys = jax.lax.dynamic_update_slice(cache.keys, k[None].astype(cache.keys.dtype), (layer, 0, pos, 0))
    values = jax.lax.dynamic_update_slice(cache.values, v[None].astype(cache.values.dtype), (layer, 0, pos, 0))
    cache = cache._replace(keys=keys, values=values)

    # Query i sees keys i - window through i, as `window_mask` does.
    seq_len = keys.shape[-2]
    window = min(attn.window or seq_len, seq_len)
    distance = (pos + jnp.arange(t))[:, None] - jnp.arange(seq_len)[None, :]
    mask = (distance >= 0) & (distance <= window)
    # `attention.scaled_dot_product_attention`, whose types want as many keys as queries.
    k, v = keys[layer].astype(q.dtype), values[layer].astype(q.dtype)
    logits = jnp.where(mask, (q @ k.swapaxes(-1, -2)) / jnp.sqrt(q.shape[-1]), -1e9)
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = weights @ v
    return mha.linear(einops.rearrange(out, "h t d -> t (h d)")), cache


def forward(
    model: decoder_only_model.DecoderOnlyTransformer,
    tokens: jt.Integer[jt.Array, " t"],
    pos: jax.Array,
    cache: Cache,
    num_valid: jax.Array | None = None,
) -> tuple[jt.Float[jt.Array, "t qkv_dim"], Cache]:
    """Hidden states for `tokens` at positions pos..pos+t, given the cache of earlier ones.

    `num_valid` says how many of `tokens` are real when the rest are padding, so that
    the cache keeps the right last embedding. Padding writes keys past the real ones,
    which later tokens overwrite before any query can see them.
    """
    config = model.config
    num_valid = tokens.shape[0] if num_valid is None else num_valid
    x = layers.rms_norm(model.embedding_layer(tokens))
    last_embedding = x[num_valid - 1]

    if config.smear:
        # The first token mixes in the cached embedding, unless it starts the sequence.
        previous = jnp.concatenate([cache.last_embedding[None].astype(x.dtype), x[:-1]], axis=0)
        channels = model.smear_gate.shape[0]
        gate = model.smear_lambda.astype(x.dtype) * jax.nn.sigmoid(
            x[..., :channels] @ model.smear_gate.astype(x.dtype)
        )
        starts = (pos + jnp.arange(tokens.shape[0]) == 0)[:, None]
        x = jnp.where(starts, x, x + gate * previous)

    embedding = x
    value_tables = iter(getattr(model, "value_embeds", []))
    for ix, block in enumerate(model.blocks):
        if config.residual_scalars:
            x = (
                model.resid_lambdas[ix].astype(x.dtype) * x
                + model.x0_lambdas[ix].astype(x.dtype) * embedding
            )
        ve = next(value_tables)(tokens) if config.has_value_embedding(ix) else None
        attended, cache = _attention(block.causal_attn, layers.rms_norm(x), ve, ix, pos, cache)
        x = x + attended
        x = x + block.mlp(layers.rms_norm(x))
        if config.backout and ix == config.num_layers // 2:
            halfway = x
    if config.backout:
        x = x - model.backout_lambda.astype(x.dtype) * halfway
    return x, cache._replace(last_embedding=last_embedding.astype(cache.last_embedding.dtype))


def _pick(logits, key, temperature, top_k):
    """The same rule as `decoder_only_model.sample`."""
    if top_k:
        kth = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0][-1]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    return jax.random.categorical(key, logits / jnp.maximum(temperature, 1e-6))


@functools.partial(jax.jit, static_argnames=("top_k",))
def _prefill(model, tokens, num_valid, cache, key, temperature, top_k):
    hidden, cache = forward(model, tokens, jnp.int32(0), cache, num_valid)
    logits = model.logits(hidden[num_valid - 1])
    return _pick(logits, key, temperature, top_k), cache


@functools.partial(jax.jit, static_argnames=("top_k",), donate_argnames=("cache",))
def _step(model, token, pos, cache, key, temperature, top_k):
    hidden, cache = forward(model, token[None], pos, cache)
    return _pick(model.logits(hidden[0]), key, temperature, top_k), cache


def _bucket(n: int) -> int:
    return max(64, 1 << (n - 1).bit_length())


@functools.partial(jax.jit, donate_argnames=("cache",))
def _prefill_batch(model, tokens, num_valid, cache):
    def one(tokens, n, cache):
        hidden, cache = forward(model, tokens, jnp.int32(0), cache, n)
        return model.logits(hidden[n - 1]).argmax(-1).astype(jnp.int32), cache

    return jax.vmap(one)(tokens, num_valid, cache)


@functools.partial(jax.jit, donate_argnames=("cache",))
def _step_batch(model, tokens, pos, cache):
    def one(token, pos, cache):
        hidden, cache = forward(model, token[None], pos, cache)
        return model.logits(hidden[0]).argmax(-1).astype(jnp.int32), cache

    return jax.vmap(one)(tokens, pos, cache)


def greedy_batch(
    model: decoder_only_model.DecoderOnlyTransformer,
    prompts: typing.Sequence[typing.Sequence[int]],
    max_new_tokens: int,
    stop_tokens: typing.Container[int] = (),
    is_done: typing.Callable[[list[int]], bool] | None = None,
    check_every: int = 16,
) -> list[list[int]]:
    """Greedy continuations of several prompts at once, for evaluations.

    Each row decodes at its own position, so prompts of different lengths share a batch.
    A row ends at a stop token (not included), after `max_new_tokens`, when the context
    is full, or once `is_done(tokens so far)` is true, checked every `check_every` steps.
    Finished rows keep computing until every row is done; their output is ignored.
    """
    seq_len = model.config.seq_len
    lengths = np.array([len(p) for p in prompts], dtype=np.int32)
    if lengths.min() < 1 or lengths.max() >= seq_len:
        raise ValueError(f"Prompts need 1 to {seq_len - 1} tokens, not {lengths.min()}..{lengths.max()}")
    width = min(_bucket(int(lengths.max())), seq_len)
    length = min(width + max_new_tokens, seq_len)
    padded = np.zeros((len(prompts), width), dtype=np.int32)
    for row, prompt in zip(padded, prompts):
        row[: len(prompt)] = prompt

    cache = empty_cache(model, length, batch=(len(prompts),))
    token, cache = _prefill_batch(model, jnp.asarray(padded), jnp.asarray(lengths), cache)
    pos, outs = lengths.copy(), [[] for _ in prompts]
    done = np.zeros(len(prompts), dtype=bool)
    for step in range(max_new_tokens):
        for i, t in enumerate(np.asarray(token)):
            if done[i]:
                continue
            if t in stop_tokens:
                done[i] = True
                continue
            outs[i].append(int(t))
            done[i] = pos[i] >= length  # The cache has no room to feed this token.
        if is_done and (step + 1) % check_every == 0:
            done |= [d or is_done(out) for d, out in zip(done, outs)]
        if done.all():
            break
        # Finished rows write past their end, clamped into the cache; nothing reads them.
        token, cache = _step_batch(model, token, jnp.asarray(np.minimum(pos, length - 1)), cache)
        pos += 1
    return outs


@dataclasses.dataclass
class Generator:
    """Streams tokens from one model. Prompts are padded to powers of two to limit compiles."""

    model: decoder_only_model.DecoderOnlyTransformer

    @property
    def seq_len(self) -> int:
        return self.model.config.seq_len

    def stream(
        self,
        prompt: typing.Sequence[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 0,
        stop_tokens: typing.Container[int] = (),
    ) -> typing.Iterator[int]:
        """Yield sampled tokens until `max_new_tokens`, a stop token, or the context is full.

        A stop token ends the stream without being yielded.
        """
        n = len(prompt)
        if not 0 < n < self.seq_len:
            raise ValueError(f"A prompt needs 1 to {self.seq_len - 1} tokens, not {n}")
        width = min(_bucket(n), self.seq_len)
        padded = np.zeros(width, dtype=np.int32)
        padded[:n] = prompt
        key = jax.random.key(seed)
        key, sub = jax.random.split(key)
        temperature = jnp.float32(temperature)
        token, cache = _prefill(self.model, jnp.asarray(padded), jnp.int32(n), empty_cache(self.model), sub, temperature, top_k)
        # `token` goes at position `pos`; the last one can fill the context but not be fed.
        pos = n
        for _ in range(max_new_tokens):
            t = int(token)
            if t in stop_tokens:
                return
            yield t
            if pos == self.seq_len:
                return
            key, sub = jax.random.split(key)
            token, cache = _step(self.model, token.astype(jnp.int32), jnp.int32(pos), cache, sub, temperature, top_k)
            pos += 1


def load(
    run_name: str,
    checkpoint_dir: str,
    step: int | None = None,
    dtype: typing.Any = None,
):
    """A trained run's model and tokenizer. `dtype` overrides the compute dtype.

    On a CPU it defaults to float32: XLA's CPU matmuls are slow in bfloat16, and d20
    decoded 29 tokens a second in float32 against 4.6 in bfloat16 on an M4 Pro.
    """
    from toylib.harness import inference
    from toylib_projects.p02_gpt_web import experiment, tokenize_webtext, train

    # Masked attention runs anywhere; the cuDNN kernel the run trained with needs a GPU.
    run = dataclasses.replace(train.RUNS[run_name], attention_impl=None)
    if dtype is None and jax.default_backend() == "cpu":
        dtype = jnp.float32
    if dtype is not None:
        run = dataclasses.replace(run, dtype=dtype)
    template = experiment.build_model(train.model_config(run), jax.random.key(0))
    model = inference.load_model(template, checkpoint_dir, step)
    tokenizer = tokenize_webtext.load_tokenizer(
        run.vocab_size, tokenize_webtext.data_dir_for(run.num_train_shards)
    )
    return model, tokenizer
