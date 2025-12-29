# ============================================================
# External Imports
# ============================================================

from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers import AutoTokenizer
import abc
import dataclasses
import datetime
import einops
import jax
import jaxtyping as jt
import json
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import pathlib
import pyarrow.parquet as pq
import typing

# ============================================================
# toylib_projects.tinystories.data - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/data.py
# ============================================================


@dataclasses.dataclass
class DatasetState:
    """Serializable state for dataset checkpointing."""

    pass


@dataclasses.dataclass
class BatchedTokenizedDataset(abc.ABC):
    dataset_path: str = "karpathy/fineweb-edu-100b-shuffle"
    split: str = "train"
    tokenizer_name: str = "gpt2"
    seq_len: int = 2048
    tokenizer_batch_size: int = 8
    batch_size: int = 128

    @abc.abstractmethod
    def _get_dataset_iterator(self) -> typing.Iterator:
        raise NotImplementedError

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.bos_token = self.tokenizer.bos_token_id
        self.token_buffer = []
        self.dataset_iter = self._get_dataset_iterator()

    def __iter__(self):
        return self

    def __next__(self) -> jnp.ndarray:
        token_needed = self.batch_size * self.seq_len + 1
        while len(self.token_buffer) < token_needed:
            input_batch = next(self.dataset_iter)
            texts = input_batch["text"]
            tokenized = self.tokenizer(
                texts,
                return_tensors=None,
                padding=False,
                truncation=False,
                max_length=None,
            )["input_ids"]
            for tokens in tokenized:
                self.token_buffer.append(self.bos_token)
                self.token_buffer.extend(tokens)
        tokens = self.token_buffer[:token_needed]
        self.token_buffer = self.token_buffer[token_needed:]
        inputs = jnp.array(tokens[:-1], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        targets = jnp.array(tokens[1:], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        return {"inputs": inputs, "targets": targets}

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing. Override in subclasses."""
        raise NotImplementedError("Checkpointing not supported for this dataset type")

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a checkpoint state. Override in subclasses."""
        raise NotImplementedError("Checkpointing not supported for this dataset type")


@dataclasses.dataclass
class DatasetStateParquet(DatasetState):
    file_index: int = 0
    row_group_index: int = 0
    token_buffer: list[int] = dataclasses.field(default_factory=list)


class BatchedTokenizedDatasetParquet(BatchedTokenizedDataset):
    """Path is constructed as dataset_path/split/*.parquet"""

    def __post_init__(self):
        self._state = DatasetStateParquet()
        super().__post_init__()

    def list_files(self) -> list[pathlib.Path]:
        base_path = pathlib.Path(self.dataset_path) / self.split
        return sorted(base_path.glob("*.parquet"))

    def _get_dataset_iterator(self) -> typing.Iterator:
        """Generator that tracks position for checkpointing."""
        files = self.list_files()
        for file_idx in range(self._state.file_index, len(files)):
            self._state.file_index = file_idx
            pf = pq.ParquetFile(files[file_idx])
            for rg_idx in range(self._state.row_group_index, pf.num_row_groups):
                self._state.row_group_index = rg_idx
                rg = pf.read_row_group(rg_idx)
                yield {"text": rg.column("text").to_pylist()}
            self._state.row_group_index = 0

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing."""
        self._state.token_buffer = self.token_buffer.copy()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore iterator position from checkpoint."""
        self._state = DatasetStateParquet(**state)
        self._state.token_buffer = state["token_buffer"].copy()
        self.dataset_iter = self._get_dataset_iterator()


# ============================================================
# toylib.nn.module - /Users/anuj/Desktop/code/toylib/toylib/nn/module.py
# ============================================================


def _is_array(x: typing.Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


class Module(abc.ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.
    """

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, kw_only=True)
        cls = jax.tree_util.register_pytree_with_keys_class(cls)

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize all the trainable parameters in the module."""
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.Any:
        """Run a forward pass of the module."""
        pass

    def _get_trainable_param_keys(self) -> list[str]:
        """Get the list of attribute names that are trainable parameters."""
        param_keys = []
        for k, v in self.__dict__.items():
            if (
                _is_array(v)
                and (not _is_random_key(k))
                or isinstance(v, Module)
                or (
                    _is_supported_container(v)
                    and all((isinstance(elem, Module) for elem in v))
                )
            ):
                param_keys.append(k)
        return param_keys

    def __post_init__(self) -> None:
        self.init()
        self._trainable_param_keys = self._get_trainable_param_keys()

    def tree_flatten_with_keys(self) -> tuple:
        params_with_keys = []
        aux_data = dict()
        for k, v in self.__dict__.items():
            if k not in self._trainable_param_keys:
                aux_data[k] = v
        for k in self._trainable_param_keys:
            v = self.__dict__[k]
            params_with_keys.append((jax.tree_util.GetAttrKey(k), v))
        return (params_with_keys, aux_data)

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> "Module":
        obj = object.__new__(cls)
        param_keys = static["_trainable_param_keys"]
        for k, v in zip(param_keys, dynamic):
            obj.__setattr__(k, v)
        for k, v in static.items():
            obj.__setattr__(k, v)
        return obj


# ============================================================
# toylib.nn.layers - /Users/anuj/Desktop/code/toylib/toylib/nn/layers.py
# ============================================================


class Linear(Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    in_features: int
    out_features: int
    use_bias: bool = False
    key: jt.PRNGKeyArray
    weights: jt.Float[jt.Array, "in_features out_features"] | None = None
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]] | None = None

    def init(self) -> None:
        w_key = self.key
        in_features = self.in_features
        out_features = self.out_features
        std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(in_features)
        self.weights = jax.random.normal(w_key, (in_features, out_features)) * std
        self.bias = jax.numpy.zeros((out_features,)) if self.use_bias else None

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class Embedding(Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    vocab_size: int
    embedding_dim: int
    key: jt.PRNGKeyArray
    weights: jt.Float[jt.Array, "vocab_size embedding_dim"] | None = None

    def init(self) -> None:
        self.weights = jax.random.normal(
            self.key, (self.vocab_size, self.embedding_dim)
        )

    def __call__(
        self, tokens: jt.Int[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0)


def rms_norm(x: jt.Float[jt.Array, "... dim"]) -> jt.Float[jt.Array, "... dim"]:
    """Applies RMS Normalization over the last dimension of the input tensor.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-09)
    return x / rms


# ============================================================
# toylib.nn.attention - /Users/anuj/Desktop/code/toylib/toylib/nn/attention.py
# ============================================================


class RotaryPositionalEmbedding(Module):
    """Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864."""

    seq_len: int = 1024
    qkv_dim: int = 128
    base: int = 100000

    def init(self) -> None:
        positions = jnp.arange(0, self.seq_len)
        freqs = self.base ** (jnp.arange(0, self.qkv_dim, 2) / self.qkv_dim)
        self.gamma = einops.einsum(positions, 1.0 / freqs, "t, d -> t d")
        self.cos = jnp.cos(self.gamma).astype(jnp.bfloat16)
        self.sin = jnp.sin(self.gamma).astype(jnp.bfloat16)

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"], t0: int = 0
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        t, d = x.shape[-2:]
        if t0 + t > self.seq_len:
            raise ValueError(
                f"Position index out of range of RoPE cache:t0 ({t0}) + t ({t}) > seq_len ({self.seq_len})"
            )
        sin, cos = (self.sin[t0 : t0 + t, :], self.cos[t0 : t0 + t, :])
        x1, x2 = (x[..., : d // 2], x[..., d // 2 :])
        es_shape = "... t d, t d -> ... t d"
        y1 = einops.einsum(x1, cos, es_shape) + einops.einsum(x2, sin, es_shape)
        y2 = -einops.einsum(x1, sin, es_shape) + einops.einsum(x2, cos, es_shape)
        return jnp.concatenate([y1, y2], axis=-1)


def scaled_dot_product_attention(
    q: jt.Float[jt.Array, "... seq_len qkv_dim"],
    k: jt.Float[jt.Array, "... seq_len qkv_dim"],
    v: jt.Float[jt.Array, "... seq_len qkv_dim"],
    mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]],
) -> tuple[
    jt.Float[jt.Array, "... seq_len qkv_dim"], jt.Float[jt.Array, "... seq_len seq_len"]
]:
    """Compute scaled dot product attention.

    Given query (`q`), key (`k`), and value (`v`) tensors, this function first computes the
    attention weights as the softmax of the dot product of `q` and `k`, scaled by the square
    root of the dimension of the keys. If a mask is provided, it is applied to the attention
    logits before the softmax is computed.

    Finally, the attention weights are used to compute the weighted average of the given values.

    NOTE: the batch dimension is not explicitly handled in this function.

    Args:
        q: query tensor
        k: keys tensor
        v: values tensor
        mask: optional boolean mask to apply to the attention logits

    Returns:
        tuple of final values and attention weights

    """
    d_k = q.shape[-1]
    assert q.shape[-1] == k.shape[-1], "q and k must have the same feature dimension"
    attention_logits = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(d_k)
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -1000000000.0)
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    values = jnp.matmul(attention_weights, v)
    return (values, attention_weights)


class MultiHeadAttention(Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    qkv_dim: int
    num_heads: int
    use_qk_norm: bool = True
    key: jt.PRNGKeyArray

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 4)
        self.q_projection = Linear(
            in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[0]
        )
        self.k_projection = Linear(
            in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[1]
        )
        self.v_projection = Linear(
            in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[2]
        )
        self.linear = Linear(in_features=qkv_dim, out_features=qkv_dim, key=keys[3])

    def __call__(
        self,
        Q: jt.Float[jt.Array, "... seq_len qkv_dim"],
        K: jt.Float[jt.Array, "... seq_len qkv_dim"],
        V: jt.Float[jt.Array, "... seq_len qkv_dim"],
        mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]] = None,
        *,
        rope: typing.Optional[RotaryPositionalEmbedding] = None,
        return_attention_weights: bool = False,
    ) -> typing.Union[
        tuple[
            jt.Float[jt.Array, "... seq_len qkv_dim"],
            jt.Float[jt.Array, "... seq_len seq_len"],
        ],
        jt.Float[jt.Array, "... seq_len qkv_dim"],
    ]:
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)
        Q = einops.rearrange(
            Q,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        K = einops.rearrange(
            K,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        V = einops.rearrange(
            V,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        if mask is not None:
            mask = einops.rearrange(
                mask, "... seq_len1 seq_len2 -> ... 1 seq_len1 seq_len2"
            )
        if rope is not None:
            Q = rope(Q)
            K = rope(K)
        if self.use_qk_norm:
            Q = rms_norm(Q)
            K = rms_norm(K)
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )
        values = einops.rearrange(
            values, "... num_heads seq_len d -> ... seq_len (num_heads d)"
        )
        values = self.linear(values)
        if return_attention_weights:
            return (values, attention_weights)
        return values


# ============================================================
# toylib_projects.tinystories.decoder_only_model - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/decoder_only_model.py
# ============================================================


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the DecoderOnlyTransformer model."""

    num_layers: int = 2
    num_heads: int = 8
    qkv_dim: int = 256
    vocab_size: int = 50257
    seq_len: int = 512
    logit_softcap: float = 15.0


class MLP(Module):
    """A simple feedforward MLP with one hidden layer."""

    qkv_dim: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 2)
        self.fc1 = Linear(in_features=qkv_dim, out_features=4 * qkv_dim, key=keys[0])
        self.fc2 = Linear(in_features=4 * qkv_dim, out_features=qkv_dim, key=keys[1])
        self.fc2.weights = jnp.zeros_like(self.fc2.weights)

    def __call__(
        self, x: jt.Float[jt.Array, "... qkv_dim"]
    ) -> jt.Float[jt.Array, "... qkv_dim"]:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


class CausalSelfAttention(Module):
    """Causal Self-Attention layer with Rotary Positional Embeddings (RoPE)."""

    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        self.mha = MultiHeadAttention(
            qkv_dim=self.qkv_dim,
            num_heads=self.num_heads,
            key=self.key,
            use_qk_norm=True,
        )
        self.mha.linear.weights = jnp.zeros_like(self.mha.linear.weights)
        self.rope = RotaryPositionalEmbedding(
            qkv_dim=self.qkv_dim // self.num_heads, seq_len=self.seq_len
        )

    def _make_causal_mask(self, seq_len: int) -> jt.Float[jt.Array, "seq_len seq_len"]:
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        x = self.mha(
            Q=x, K=x, V=x, mask=self._make_causal_mask(x.shape[-2]), rope=self.rope
        )
        return x


class DecoderBlock(Module):
    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.causal_attn = CausalSelfAttention(
            qkv_dim=self.qkv_dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            key=keys[0],
        )
        self.mlp = MLP(qkv_dim=self.qkv_dim, key=keys[1])

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        x = x + self.causal_attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class DecoderOnlyTransformer(Module):
    """A simple decoder-only transformer model.

    Takes in a sequence of tokens, embeds them using a learned embedding layer,
    applies causal self-attention with ROPE embeddings, and outputs the logits
    for the next token prediction.
    """

    key: jt.PRNGKeyArray
    config: ModelConfig

    def init(self) -> None:
        config = self.config
        keys = jax.random.split(self.key, config.num_layers + 2)
        self.embedding_layer = Embedding(
            vocab_size=config.vocab_size, embedding_dim=config.qkv_dim, key=keys[0]
        )
        self.blocks = []
        for ix in range(config.num_layers):
            self.blocks.append(
                DecoderBlock(
                    qkv_dim=config.qkv_dim,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len,
                    key=keys[ix + 1],
                )
            )
        self.output_layer = Linear(
            in_features=config.qkv_dim, out_features=config.vocab_size, key=keys[-1]
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len vocab_size"]:
        """Forward pass for the decoder-only transformer model.

        Args:
            x: Input token ids of shape [seq_len]. Note that the sequence
                length should match the configuration `seq_len` as this is used for
                positional embeddings.

        Returns:
            Unnormalized logits over the vocabulary of shape [batch_size, seq_len, vocab_size]
        """
        x = self.embedding_layer(x)
        x = rms_norm(x)
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x)
        x = self.output_layer(x)
        x = self.config.logit_softcap * jnp.tanh(x / self.config.logit_softcap)
        return x


def loss_fn(
    logits: jt.Float[jt.Array, "batch_size seq_len vocab_size"],
    targets: jt.Int[jt.Array, "batch_size seq_len"],
    mask: jt.Int[jt.Array, "batch_size seq_len"],
) -> jt.Float[jt.Array, ""]:
    """Computes the cross-entropy loss between logits and targets.

    Args:
        logits: Logits of shape [batch_size, seq_len, vocab_size].
        targets: Target token ids of shape [batch_size, seq_len].

    Returns:
        Scalar loss value.
    """
    targets_one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_token_loss = -jnp.sum(targets_one_hot * log_probs, axis=-1)
    masked_loss = mask * per_token_loss
    total_loss = jnp.sum(masked_loss) / jnp.sum(mask)
    return (total_loss, per_token_loss)


def train_step(
    model: DecoderOnlyTransformer,
    tokens: jt.Int[jt.Array, "batch_size seq_len"],
    mask: jt.Int[jt.Array, "batch_size seq_len"],
    targets: jt.Int[jt.Array, "batch_size seq_len"],
) -> jt.Float[jt.Array, ""]:
    """A single training step for the model.

    Args:
        model: The DecoderOnlyTransformer model.
        batch: Input token ids of shape [batch_size, seq_len].
        labels: Target token ids of shape [batch_size, seq_len].

    Returns:
        Loss value for the batch.
    """
    logits = model(tokens)
    total_loss, per_token_loss = loss_fn(logits, targets, mask)
    return (total_loss, {"logits": logits, "per_token_loss": per_token_loss})


def sample(
    model: DecoderOnlyTransformer,
    input_tokens: list[int],
    key: jt.PRNGKeyArray,
    *,
    max_output_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> jt.Int[jt.Array, "batch_size seq_len"]:
    """Generates samples from the model given input tokens.

    This is a simple implementation with no optimizations or batching support.

    Args:
        model: The DecoderOnlyTransformer model.
        input_tokens: Input token ids of shape [seq_len].
        key: JAX PRNG key for sampling.
        max_output_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: If specified, only consider the top_k logits for sampling. Set
            to 1 for greedy sampling.

    Yields:
        Generated token ids one at a time.
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")
    tokens = input_tokens.copy()
    for _ in range(max_output_tokens):
        outputs = model(jnp.array(tokens))
        logits = outputs[-1, :]
        logits /= temperature
        if top_k:
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            logits = jnp.where(logits < top_k_logits[-1], -jnp.inf, logits)
        next_token = jax.random.categorical(key=key, logits=logits)
        tokens.append(next_token)
        yield next_token


# ============================================================
# toylib_projects.tinystories.logger - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/logger.py
# ============================================================


class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    @abc.abstractmethod
    def log(self, step: int, metrics: dict) -> None:
        """Log the given metrics at the specified step."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close any resources held by the logger."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(self, config_dict: dict, output_path: str, *args, **kwargs) -> None:
        self.config_dict = config_dict
        self.file_ptr = open(output_path, "w")
        self.file_ptr.write("\n")

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["timestamp"] = timestamp
        metrics["step"] = step
        self.file_ptr.write(json.dumps(metrics) + "\n")
        self.file_ptr.flush()

    def close(self) -> None:
        self.file_ptr.close()


class StdoutLogger(Logger):
    """Logger implementation that logs metrics to standard output."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Step {step}: {metrics}")

    def close(self) -> None:
        pass


# ============================================================
# toylib_projects.tinystories.experiment - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/experiment.py
# ============================================================

"""Basic types for the training loop and configurations."""
DEFAULT_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


@dataclasses.dataclass
class CheckpointConfig:
    save_interval_steps: int = 5000
    max_to_keep: typing.Optional[int] = 10
    checkpoint_dir: str = "/tmp/checkpoints"
    checkpoint_dataset_iterator: bool = False


@dataclasses.dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    max_steps: int = 100000


@dataclasses.dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    num_eval_steps: int = 1


@dataclasses.dataclass
class Task:
    name: str
    dataset: BatchedTokenizedDataset


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: Logger = FileLogger
    log_dir: str = "/tmp/train_logs.txt"
    train_log_interval_steps: int = 1


def _serialize_dataclass_config(config: dataclasses.dataclass) -> dict:
    result = dataclasses.asdict(config)
    for k, v in result.items():
        if dataclasses.is_dataclass(v):
            result[k] = _serialize_dataclass_config(v)
    return result


@dataclasses.dataclass
class Experiment:
    """Base Experiment class."""

    train_task: Task
    eval_task: Task | None = None
    model_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    eval_config: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    checkpoint_config: CheckpointConfig = dataclasses.field(
        default_factory=CheckpointConfig
    )
    logger_config: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)
    forward_fn: ... = dataclasses.field(default_factory=lambda: train_step)
    jit_computations: bool = True

    def __post_init__(self):
        self.num_devices = jax.local_device_count()
        devices = np.array(jax.local_devices())
        self.mesh = Mesh(devices, axis_names=("data",))
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.data_sharding = NamedSharding(self.mesh, P("data"))
        train_batch_size = self.train_task.dataset.batch_size
        eval_batch_size = (
            self.eval_task.dataset.batch_size if self.eval_task is not None else 0
        )
        if train_batch_size % self.num_devices != 0:
            raise ValueError(
                f"Batch size {self.batch_size} not divisible by number of devices {self.num_devices}"
            )
        if eval_batch_size % self.num_devices != 0 and eval_batch_size != 0:
            raise ValueError(
                f"Eval batch size {eval_batch_size} not divisible by number of devices {self.num_devices}"
            )
        print(
            f"Initialized mesh {self.mesh} with {self.num_devices} devices: {devices}"
        )
        self.logger_obj = self.logger_config.logger_cls(
            config_dict=_serialize_dataclass_config(self),
            output_path=self.logger_config.log_dir,
        )
        self.optimizer = optax.adam(learning_rate=self.training_config.learning_rate)
        self.opt_state = None
        self.model = None
        self.ckpt_manager = ocp.CheckpointManager(
            self.checkpoint_config.checkpoint_dir,
            checkpointers={
                "model": ocp.StandardCheckpointer(),
                "opt_state": ocp.StandardCheckpointer(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.checkpoint_config.max_to_keep
            ),
        )

        def train_step(model, opt_state, batch):
            inputs, targets = (batch["inputs"], batch["targets"])
            mask = jnp.ones_like(inputs)
            with jax.profiler.TraceAnnotation("value_and_grad"):
                (loss_val, _), grads = jax.value_and_grad(
                    self.forward_fn, has_aux=True
                )(model, inputs, mask, targets)
                grads = jax.tree.map(lambda g: g / self.num_devices, grads)
            with jax.profiler.TraceAnnotation("optimizer_update"):
                updates, opt_state = self.optimizer.update(grads, opt_state)
                model = optax.apply_updates(model, updates)
            return (model, opt_state, loss_val)

        def eval_step(model, batch):
            inputs, targets = (batch["inputs"], batch["targets"])
            mask = jnp.ones_like(inputs)
            with jax.profiler.TraceAnnotation("eval_forward"):
                loss_val, _ = self.forward_fn(model, inputs, mask, targets)
            return loss_val

        if self.jit_computations:
            self.train_step_fn = jax.jit(
                train_step,
                in_shardings=(
                    self.replicated_sharding,
                    self.replicated_sharding,
                    self.data_sharding,
                ),
                out_shardings=(
                    self.replicated_sharding,
                    self.replicated_sharding,
                    self.replicated_sharding,
                ),
            )
            self.eval_step_fn = jax.jit(
                eval_step,
                in_shardings=(self.replicated_sharding, self.data_sharding),
                out_shardings=self.replicated_sharding,
            )
        else:
            self.train_step_fn = train_step
            self.eval_step_fn = eval_step

    def init_state(self):
        self.model = DecoderOnlyTransformer(
            config=self.model_config, key=jax.random.PRNGKey(0)
        )
        self.model = jax.device_put(self.model, self.replicated_sharding)
        self.opt_state = self.optimizer.init(self.model)
        self.opt_state = jax.device_put(self.opt_state, self.replicated_sharding)
        self.step = 0
        print(f"Model initialized and replicated across {self.num_devices} devices")

    def _assert_initialized(self) -> bool:
        initialized = self.model is not None and self.opt_state is not None
        assert initialized, "Experiment state not initialized. Call init_state() first."

    def _unreplicate_for_checkpoint(self, pytree):
        """Get a single copy of replicated state for checkpointing."""
        return jax.tree.map(lambda x: np.asarray(x), pytree)

    def save_checkpoint(self):
        self._assert_initialized()
        model_to_save = self._unreplicate_for_checkpoint(self.model)
        opt_state_to_save = self._unreplicate_for_checkpoint(self.opt_state)
        args = {
            "model": ocp.args.StandardSave(model_to_save),
            "opt_state": ocp.args.StandardSave(opt_state_to_save),
        }
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardSave(
                self.train_task.dataset.get_state()
            )
        self.ckpt_manager.save(self.step, args=ocp.args.Composite(**args))
        self.ckpt_manager.wait_until_finished()

    def restore_checkpoint(self, step: int):
        self._assert_initialized()
        model_template = self._unreplicate_for_checkpoint(self.model)
        opt_state_template = self._unreplicate_for_checkpoint(self.opt_state)
        args = {
            "model": ocp.args.StandardRestore(model_template),
            "opt_state": ocp.args.StandardRestore(opt_state_template),
        }
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardRestore(
                self.train_task.dataset.get_state()
            )
        restored = self.ckpt_manager.restore(step, args=ocp.args.Composite(**args))
        self.model = jax.device_put(restored["model"], self.replicated_sharding)
        self.opt_state = jax.device_put(restored["opt_state"], self.replicated_sharding)
        if self.checkpoint_config.checkpoint_dataset_iterator:
            self.train_task.dataset.restore_state(restored["dataset_iterator"])
        self.step = step

    def run_validation(self) -> float:
        self._assert_initialized()
        if self.eval_task is None:
            print("No eval task defined, skipping validation loss.")
            return
        total_val_loss = 0.0
        for ix, batch in enumerate(self.eval_task.dataset):
            val_loss = self.eval_step_fn(self.model, batch)
            total_val_loss += float(val_loss)
            if ix >= self.eval_config.num_eval_steps:
                break
        avg_val_loss = total_val_loss / (ix + 1)
        self.logger_obj.log(self.step, metrics={"val/loss": avg_val_loss})
        return avg_val_loss

    def sampling_evaluation(
        self, prompts: list[str] | None = None, max_tokens: int = 10
    ) -> None:
        """Run sampling evaluation (runs on single device for simplicity).

        Args:
            prompts: List of string prompts to evaluate.
        """
        self._assert_initialized()
        if prompts is None:
            prompts = DEFAULT_PROMPTS
        model_single = jax.tree.map(lambda x: np.asarray(x), self.model)
        results = []
        tokenized_prompts = self.train_task.dataset.tokenizer(
            prompts,
            return_tensors=None,
            padding=False,
            truncation=False,
            max_length=None,
        )["input_ids"]
        for ix in range(len(tokenized_prompts)):
            generated = list(
                sample(
                    model=model_single,
                    input_tokens=tokenized_prompts[ix],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=max_tokens,
                    temperature=1.0,
                    top_k=5,
                )
            )
            results.append(
                {
                    "prompt": prompts[ix],
                    "output": self.train_task.dataset.tokenizer.decode(generated),
                }
            )
        self.logger_obj.log(self.step, metrics={"step": self.step, "samples": results})

    def eval(self):
        self._assert_initialized()
        self.run_validation()
        self.sampling_evaluation()

    def inner_loop(self, batch: dict):
        self._assert_initialized()
        self.model, self.opt_state, loss_val = self.train_step_fn(
            self.model, self.opt_state, batch
        )
        if self.step % self.logger_config.train_log_interval_steps == 0:
            self.logger_obj.log(self.step, metrics={"train/loss": float(loss_val)})

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps
        while True:
            epoch_start_step = self.step
            for batch in self.train_task.dataset:
                with jax.profiler.StepTraceAnnotation("inner_loop", step_num=self.step):
                    self.inner_loop(batch)
                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()
                if self.step % self.eval_config.eval_interval_steps == 0:
                    self.eval()
                self.step += 1
                if self.step >= self.training_config.max_steps:
                    finished = True
                    break
            if finished:
                break
            if self.step == epoch_start_step:
                raise ValueError(f"Dataset for task {self.train_task.name} is empty.")

    def cleanup(self):
        self.logger_obj.close()
        self.ckpt_manager.close()


# ============================================================
# None - ../tinystories/train.py
# ============================================================


def get_model_config(
    depth: int, seq_len: int = 1024, vocab_size: int = 50257
) -> ModelConfig:
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    print(f"num_layers: {num_layers}")
    print(f"model_dim: {model_dim}")
    print(f"num_heads: {num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    return ModelConfig(
        num_layers=depth,
        num_heads=num_heads,
        qkv_dim=model_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )


def create_experiment(
    batch_size: int = 8,
    seq_len: int = 2048,
    depth: int = 12,
    vocab_size: int = 50257,
    checkpoint_dir: str = "/tmp/checkpoints",
    dataset_path: str = "/tmp/",
    dataset_train_split: str = "train",
    dataset_val_split: str | None = "val",
) -> Experiment:
    train_task = Task(
        name="train",
        dataset=BatchedTokenizedDatasetParquet(
            dataset_path=dataset_path,
            split=dataset_train_split,
            batch_size=batch_size,
            seq_len=seq_len,
            tokenizer_batch_size=8,
        ),
    )
    val_task = None
    if dataset_val_split is not None:
        val_task = Task(
            name="val",
            dataset=BatchedTokenizedDatasetParquet(
                dataset_path=dataset_path,
                split=dataset_val_split,
                batch_size=batch_size,
                seq_len=seq_len,
                tokenizer_batch_size=8,
            ),
        )
    exp = Experiment(
        model_config=get_model_config(
            depth=depth, seq_len=seq_len, vocab_size=vocab_size
        ),
        training_config=TrainingConfig(learning_rate=0.001, max_steps=100000),
        checkpoint_config=CheckpointConfig(
            save_interval_steps=2500,
            max_to_keep=10,
            checkpoint_dir=checkpoint_dir,
            save_dataset_iterator=False,
        ),
        train_task=train_task,
        eval_task=val_task,
    )
    return exp


if __name__ == "__main__":
    exp = create_experiment(batch_size=48)
    exp.init_state()
    exp.outer_loop()
