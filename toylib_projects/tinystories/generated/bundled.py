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
import grain.python as grain
import jax
import jaxtyping as jt
import json
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import pandas as pd
import pathlib
import pyarrow.parquet as pq
import typing

# ============================================================
# toylib_projects.tinystories.analyze - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/analyze.py
# ============================================================


def get_tree_stats(model: jt.PyTree) -> pd.DataFrame:
    """
    Groups parameter counts and MiB sizes at a specified depth.
    """
    results = []
    leaf_stats = [
        (k, v.shape, v.dtype) for k, v in jax.tree_util.tree_leaves_with_path(model)
    ]
    for path, shape, dtype in leaf_stats:
        path = [str(p) for p in path]
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        results.append({"params": count, "n_bytes": nbytes, "path": "/".join(path)})
        for i, p in enumerate(path):
            results[-1][f"level_{i}"] = p
    return pd.DataFrame(results)


def print_param_sizes(
    model: jt.PyTree, depth: int = 1, size_denom: int = 1
) -> tuple[pd.DataFrame, int, int]:
    """
    Analyzes parameters and the compiled XLA HLO for peak memory usage.
    """
    df_stats = get_tree_stats(model)
    if len(df_stats) == 0:
        print("Model has no parameters.")
        return pd.DataFrame()
    df_stats.loc[:, "n_bytes_divided"] = df_stats["n_bytes"] / size_denom
    total_params = df_stats.params.sum()
    total_bytes = df_stats.n_bytes_divided.sum()
    print(f"Total Parameters: {total_params}. Bytes: ({total_bytes:.2f})")
    return (
        df_stats.fillna("")
        .groupby([f"level_{i}" for i in range(depth)])
        .sum()[["params", "n_bytes_divided"]]
        .reset_index(),
        total_params,
        total_bytes,
    )


def print_xla_memory_analysis(
    train_step_fn: typing.Callable,
    params: jt.PyTree,
    batch: typing.Mapping[str, jt.Array],
):
    lowered = jax.jit(train_step_fn).lower(params, batch)
    compiled = lowered.compile()
    analysis = compiled.memory_analysis()

    def _to_mib(b: int) -> float:
        return b / 1024**2

    print("\n--- XLA Compilation Estimate ---")
    print(
        f"Arguments (Params + Batch):\t{_to_mib(analysis.argument_size_in_bytes):.2f} MiB"
    )
    print(f"Output (Grads + Loss):\t{_to_mib(analysis.output_size_in_bytes):.2f} MiB")
    print(f"Temp/Activations (Peak):\t{_to_mib(analysis.temp_size_in_bytes):.2f} MiB")
    print(
        f"Total Peak Memory:\t{_to_mib(analysis.temp_size_in_bytes + analysis.argument_size_in_bytes):.2f} MiB"
    )


def print_estimated_tokens(exp) -> int:
    """Estimate total number of tokens processed during training."""
    total_tokens = (
        exp.training_config.max_steps
        * exp.train_task.dataset.batch_size
        * exp.train_task.dataset.seq_len
    )
    print("------------------------------")
    print("Token Analysis:")
    print("------------------------------")
    print("Batch size:", exp.train_task.dataset.batch_size)
    print("Seq len:", exp.train_task.dataset.seq_len)
    print("Max steps:", exp.training_config.max_steps)
    print(
        "Num microbatches (split from within batch_size):",
        exp.training_config.num_microbatches,
    )
    print("Total training tokens:", total_tokens)
    print("------------------------------")


def print_chinchilla_estimate(model: jt.PyTree):
    _, model_params, _ = print_param_sizes(model, depth=2, size_denom=1)
    print("------------------------------")
    print("Chinchilla Analysis:")
    print("------------------------------")
    print("Model parameters:", model_params)
    print("Chinchilla estimate: (20 * model_params):", 20 * model_params, "tokens")
    print("------------------------------")


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
        return {"inputs": inputs, "targets": targets, "mask": jnp.ones_like(inputs)}

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


@dataclasses.dataclass
class DatasetStateGrain(DatasetState):
    """State for Grain-based dataset checkpointing."""

    sampler_state: dict = dataclasses.field(default_factory=dict)
    token_buffer: list[int] = dataclasses.field(default_factory=list)


class BatchedTokenizedDatasetGrain(BatchedTokenizedDataset):
    """Grain-based data loader that reads parquet files.

    Uses Grain's efficient data loading pipeline for reading parquet files.
    Path is constructed as dataset_path/split/*.parquet

    Example:
        >>> dataset = BatchedTokenizedDatasetGrain(
        ...     dataset_path="/path/to/data",
        ...     split="train",
        ...     batch_size=128,
        ...     seq_len=2048,
        ... )
        >>> for batch in dataset:
        ...     train_step(batch)
    """

    seed: int = 42

    def __post_init__(self):
        self._state = DatasetStateGrain()
        super().__post_init__()

    def list_files(self) -> list[pathlib.Path]:
        """List parquet files for the split."""
        base_path = pathlib.Path(self.dataset_path) / self.split
        return sorted(base_path.glob("*.parquet"))

    def _get_dataset_iterator(self) -> typing.Iterator:
        """Create Grain-based iterator over parquet files."""
        dataset = grain.MapDataset.source([str(f) for f in self.list_files()])
        dataset = dataset.map(grain.experimental.ParquetIterDataset)
        dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=1)
        dataset = dataset.batch(self.tokenizer_batch_size, drop_remainder=False)
        iterator = iter(dataset)
        if self._state.sampler_state:
            iterator.set_state(self._state.sampler_state)
        self._grain_iterator = iterator
        for batch in iterator:
            yield {"text": list(batch["text"])}

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing.

        Returns:
            Dictionary containing iterator state and token buffer.
        """
        self._state.sampler_state = self._grain_iterator.get_state()
        self._state.token_buffer = self.token_buffer.copy()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a checkpoint state.

        Args:
            state: State dictionary containing sampler state and token buffer.
        """
        self._state = DatasetStateGrain(**state)
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
        self, tokens: jt.Integer[jt.Array, "... seq_len"]
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


@dataclasses.dataclass(frozen=True)
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
        self, x: jt.Integer[jt.Array, "... seq_len"]
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
    model: DecoderOnlyTransformer, batch: jt.PyTree
) -> jt.Float[jt.Array, ""]:
    """A single training step for the model.

    Args:
        model: The DecoderOnlyTransformer model.
        batch: PyTree containing 'inputs', 'targets', and 'mask', each of shape
            [batch_size, seq_len].

    Returns:
        Loss value for the batch.
    """
    tokens, targets, mask = (batch["inputs"], batch["targets"], batch["mask"])
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
        self.file_ptr = open(os.path.join(output_path, "train_logs.txt"), "w")
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
# toylib_projects.tinystories.metrics - /Users/anuj/Desktop/code/toylib/toylib_projects/tinystories/metrics.py
# ============================================================


class Metric(typing.Protocol):
    """Protocol for computing and accumulating metrics."""

    def __call__(
        self, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        """Compute final metric value(s) for the given inputs.

        Args:
            loss: The loss value returned by forward_fn
            aux: The auxiliary jt.PyTree returned by forward_fn
            batch: The input batch
        """
        pass


@dataclasses.dataclass
class Loss:
    """Pass-through metric that returns the loss value."""

    def __call__(
        self, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        """Return the loss value.

        Args:
            loss: The loss value returned by forward_fn
            aux: The auxiliary PyTree (unused)
            batch: The input batch (unused)

        Returns:
            Dictionary with 'loss' metric
        """
        del aux, batch
        return {"loss": loss}


@dataclasses.dataclass
class BitsPerByte:
    """Metric that computes bits per byte from per-token loss.

    This metric converts the per-token loss (in nats) to bits per byte by:
    1. Converting nats to bits (multiply by log2(e))
    2. Dividing by the number of bytes per token for each token

    The bytes per token mapping is loaded from an .npy file at initialization.
    """

    bytes_per_token_path: str
    _bytes_per_token: jt.Array = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """Load the bytes per token array from disk."""
        self._bytes_per_token = jnp.array(np.load(self.bytes_per_token_path))

    def __call__(
        self, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        """Compute bits per byte metric.

        Args:
            loss: The loss value (unused)
            aux: Must contain 'per_token_loss' with shape [batch_size, seq_len]
            batch: Must contain 'inputs' with token ids of shape [batch_size, seq_len]

        Returns:
            Dictionary with 'bits_per_byte' metric
        """
        del loss
        per_token_loss = aux["per_token_loss"]
        token_ids = batch["inputs"]
        mask = batch["mask"]
        bytes_per_token = self._bytes_per_token[token_ids]
        bits_per_token = per_token_loss * jnp.log2(jnp.e)
        bits_per_byte = bits_per_token / bytes_per_token
        token_valid = jnp.where(bytes_per_token == -1, 0, 1) * mask
        mean_bits_per_byte = (bits_per_byte * token_valid).sum() / (
            token_valid.sum() + 1e-19
        )
        return {"bits_per_byte": mean_bits_per_byte}


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
class OptimizerConfig:
    """Configuration for a single optimizer."""

    name: str
    optimizer: optax.GradientTransformation


@dataclasses.dataclass
class MultiOptimizerConfig:
    """Configuration for multi-optimizer training."""

    optimizer_configs: list[OptimizerConfig]
    optimizer_for_param: typing.Callable[[tuple], str]

    def build_optimizer_map(self) -> dict[str, optax.GradientTransformation]:
        if not self.optimizer_configs:
            raise ValueError("multi_optimizer_config.optimizer_configs cannot be empty")
        optimizer_map = {
            config.name: config.optimizer for config in self.optimizer_configs
        }
        assert len(optimizer_map) == len(self.optimizer_configs)
        return optimizer_map


@dataclasses.dataclass
class TrainingConfig:
    optimizer_config: MultiOptimizerConfig | None = None
    max_steps: int = 100000
    num_microbatches: int = 1
    max_grad_norm: float = 0.0


@dataclasses.dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    num_eval_steps: int = 1


@dataclasses.dataclass
class Task:
    name: str
    dataset: BatchedTokenizedDataset
    metrics: list[Metric] = dataclasses.field(default_factory=lambda: [Loss()])


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: Logger = FileLogger
    log_dir: str = "/tmp/"
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

    def _validate_configs(self) -> None:
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
        if (
            train_batch_size / self.num_devices % self.training_config.num_microbatches
            != 0
        ):
            raise ValueError(
                f"Number of microbatches {self.training_config.num_microbatches} does not evenly divide per-device batch size {train_batch_size // self.num_devices}"
            )

    def _setup_sharding(self) -> None:
        self.num_devices = jax.local_device_count()
        devices = np.array(jax.local_devices())
        self.mesh = Mesh(devices, axis_names=("data",))
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.data_sharding = NamedSharding(self.mesh, P("data"))
        print(
            f"Initialized mesh {self.mesh} with {self.num_devices} devices: {devices}"
        )

    def _compute_metrics(
        self, task: Task, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        """Compute all metrics for a task.

        Args:
            task: The task containing the metrics to compute
            loss: The loss value
            aux: The auxiliary PyTree from forward_fn
            batch: The input batch

        Returns:
            Dictionary mapping metric names to values
        """
        all_metrics = {}
        for metric in task.metrics:
            metric_results = metric(loss=loss, aux=aux, batch=batch)
            all_metrics.update(metric_results)
        return all_metrics

    def _create_optimizer(self, model: jt.PyTree) -> optax.GradientTransformation:
        """Create the optimizer with optional per-parameter optimization.

        This is called after model initialization to create an optimizer that
        can apply different optimizers to different parts of the model. The
        gradient clipping is applied globally before the multi-optimizer.

        Args:
            model: The initialized model PyTree

        Returns:
            Optax optimizer chain with gradient clipping and multi-optimizer
        """
        optimizer_chain = []
        if self.training_config.max_grad_norm > 0.0:
            optimizer_chain.append(
                optax.clip_by_global_norm(self.training_config.max_grad_norm)
            )
        if self.training_config.optimizer_config is None:
            print("Using default optimizer: Adam, 1e-3")
            optimizer_chain.append(optax.adam(learning_rate=0.001))
        else:
            optimizer_map = self.training_config.optimizer_config.build_optimizer_map()
            optimizer_for_param = (
                self.training_config.optimizer_config.optimizer_for_param
            )

            def label_fn(params):
                """Map params PyTree to labels PyTree using optimizer_for_param."""
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: optimizer_for_param(path), params
                )

            optimizer_chain.append(
                optax.multi_transform(transforms=optimizer_map, param_labels=label_fn)
            )
        if len(optimizer_chain) == 1:
            return optimizer_chain[0]
        return optax.chain(*optimizer_chain)

    def _train_step(self, model, opt_state, batch):
        """Perform a single training step with microbatching."""

        def _slice_tensors(
            batch: jt.PyTree, start: int, microbatch_size: int
        ) -> jt.PyTree:
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, start, microbatch_size, axis=0
                ),
                batch,
            )

        sharded_batch_size = batch["inputs"].shape[0]
        num_microbatches = self.training_config.num_microbatches
        microbatch_size = sharded_batch_size // num_microbatches
        total_grads = jax.tree.map(lambda x: jnp.zeros_like(x), model)
        accumulated_metrics = None
        with jax.profiler.TraceAnnotation("microbatch_loop"):
            for microbatch_idx in range(num_microbatches):
                start_idx = microbatch_idx * microbatch_size
                microbatch = _slice_tensors(batch, start_idx, microbatch_size)
                (loss_val, aux), grads = jax.value_and_grad(
                    self.forward_fn, has_aux=True
                )(model, microbatch)
                microbatch_metrics = self._compute_metrics(
                    task=self.train_task, loss=loss_val, aux=aux, batch=microbatch
                )
                total_grads = jax.tree.map(lambda x, y: x + y, total_grads, grads)
                if accumulated_metrics is None:
                    accumulated_metrics = microbatch_metrics
                else:
                    accumulated_metrics = jax.tree.map(
                        lambda x, y: x + y, accumulated_metrics, microbatch_metrics
                    )
            total_grads = jax.tree.map(
                lambda g: g / self.num_devices / num_microbatches, total_grads
            )
            averaged_metrics = jax.tree.map(
                lambda x: x / num_microbatches, accumulated_metrics
            )
        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.optimizer.update(total_grads, opt_state, model)
            model = optax.apply_updates(model, updates)
        return (model, opt_state, averaged_metrics)

    def _eval_step(self, model, batch):
        """Perform a single evaluation step and compute metrics."""
        with jax.profiler.TraceAnnotation("eval_forward"):
            loss_val, aux = self.forward_fn(model, batch)
        eval_metrics = self._compute_metrics(
            task=self.eval_task, loss=loss_val, aux=aux, batch=batch
        )
        return eval_metrics

    def __post_init__(self):
        self._setup_sharding()
        self._validate_configs()
        self.logger_obj = self.logger_config.logger_cls(
            config_dict=_serialize_dataclass_config(self),
            output_path=self.logger_config.log_dir,
        )
        self.optimizer = None
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
        if self.jit_computations:
            self.train_step_fn = jax.jit(
                self._train_step,
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
                self._eval_step,
                in_shardings=(self.replicated_sharding, self.data_sharding),
                out_shardings=self.replicated_sharding,
            )
        else:
            self.train_step_fn = self._train_step
            self.eval_step_fn = self._eval_step

    def init_state(self):
        self.model = DecoderOnlyTransformer(
            config=self.model_config, key=jax.random.key(0)
        )
        self.model = jax.device_put(self.model, self.replicated_sharding)
        self.optimizer = self._create_optimizer(self.model)
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

    def run_validation(self) -> dict[str, float]:
        """Run validation and compute all metrics for the eval task.

        Returns:
            Dictionary of averaged metrics
        """
        self._assert_initialized()
        if self.eval_task is None:
            print("No eval task defined, skipping validation.")
            return {}
        accumulated_metrics = None
        num_batches = 0
        for ix, batch in enumerate(self.eval_task.dataset):
            batch_metrics = self.eval_step_fn(self.model, batch)
            if accumulated_metrics is None:
                accumulated_metrics = batch_metrics
            else:
                accumulated_metrics = jax.tree.map(
                    lambda x, y: x + y, accumulated_metrics, batch_metrics
                )
            num_batches += 1
            if ix >= self.eval_config.num_eval_steps:
                break
        avg_metrics = jax.tree.map(
            lambda x: float(x) / num_batches, accumulated_metrics
        )
        avg_metrics = {f"val/{key}": value for key, value in avg_metrics.items()}
        self.logger_obj.log(self.step, metrics=avg_metrics)
        return avg_metrics

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
                    key=jax.random.key(0),
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
        self.model, self.opt_state, train_metrics = self.train_step_fn(
            self.model, self.opt_state, batch
        )
        if self.step % self.logger_config.train_log_interval_steps == 0:
            train_metrics_with_prefix = {
                f"train/{key}": float(value) for key, value in train_metrics.items()
            }
            self.logger_obj.log(self.step, metrics=train_metrics_with_prefix)

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


def create_muon_adam_multi_optimizer_config(
    muon_lr: float = 0.0001,
    adamw_embed_lr: float = 0.0001,
    adamw_output_lr: float = 0.0001,
    weight_decay: float = 0.0,
) -> MultiOptimizerConfig:
    """Create multi-optimizer config with Muon for blocks, Adam for embeddings/output.

    Optimizer routing:
    - embedding_layer -> Adam (embeddings typically need different treatment)
    - output_layer -> Adam (output projection)
    - everything else (blocks with causal_attn and mlp) -> Muon

    Args:
        muon_lr: Learning rate for Muon optimizer
        adamw_embed_lr: Learning rate for Adam optimizer (embeddings)
        adamw_output_lr: Learning rate for Adam optimizer (output)

    Returns:
        MultiOptimizerConfig ready for TrainingConfig
    """

    def optimizer_for_param(key_path: tuple) -> str:
        """Route parameters to optimizers based on their path in the model tree."""
        path_strs = []
        for k in key_path:
            if hasattr(k, "key"):
                path_strs.append(k.key if isinstance(k.key, str) else str(k.key))
            else:
                path_strs.append(str(k))
        if "embedding_layer" in path_strs:
            return "adamw_embed"
        if "output_layer" in path_strs:
            return "adamw_output"
        return "muon"

    optimizer_configs = [
        OptimizerConfig(
            name="muon", optimizer=optax.contrib.muon(learning_rate=muon_lr)
        ),
        OptimizerConfig(
            name="adamw_embed",
            optimizer=optax.adamw(
                learning_rate=adamw_embed_lr,
                b1=0.8,
                b2=0.95,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
        ),
        OptimizerConfig(
            name="adamw_output",
            optimizer=optax.adamw(
                learning_rate=adamw_output_lr,
                b1=0.8,
                b2=0.95,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
        ),
    ]
    return MultiOptimizerConfig(
        optimizer_configs=optimizer_configs, optimizer_for_param=optimizer_for_param
    )


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
    batch_size_per_device: int = 18,
    seq_len: int = 2048,
    max_steps: int = 12000,
    num_microbatches: int = 2,
    depth: int = 12,
    vocab_size: int = 50257,
    checkpoint_dir: str = "/tmp/checkpoints",
    dataset_path: str = "/tmp/",
    dataset_train_split: str = "train",
    dataset_val_split: str | None = "val",
    bpt_path: str = "/tmp/bpt_gpt2.npy",
    muon_lr: float = 0.02,
    adamw_embed_lr: float = 0.2,
    adamw_output_lr: float = 0.004,
) -> Experiment:
    batch_size = batch_size_per_device * jax.local_device_count() * num_microbatches
    train_task = Task(
        name="train",
        dataset=BatchedTokenizedDatasetGrain(
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
            dataset=BatchedTokenizedDatasetGrain(
                dataset_path=dataset_path,
                split=dataset_val_split,
                batch_size=batch_size,
                seq_len=seq_len,
                tokenizer_batch_size=8,
            ),
            metrics=[Loss(), BitsPerByte(bpt_path)],
        )
    model_config = get_model_config(depth=depth, seq_len=seq_len, vocab_size=vocab_size)
    optimizer_config = None
    model_dim = model_config.qkv_dim
    dmodel_lr_scale = (model_dim / 768) ** (-0.5)
    print(
        f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
    )
    optimizer_config = create_muon_adam_multi_optimizer_config(
        muon_lr=muon_lr,
        adamw_embed_lr=adamw_embed_lr * dmodel_lr_scale,
        adamw_output_lr=adamw_output_lr * dmodel_lr_scale,
    )
    exp = Experiment(
        model_config=model_config,
        training_config=TrainingConfig(
            max_steps=max_steps,
            num_microbatches=num_microbatches,
            max_grad_norm=1.0,
            optimizer_config=optimizer_config,
        ),
        checkpoint_config=CheckpointConfig(
            save_interval_steps=2500,
            max_to_keep=10,
            checkpoint_dir=checkpoint_dir,
            checkpoint_dataset_iterator=False,
        ),
        logger_config=LoggerConfig(log_dir=checkpoint_dir),
        train_task=train_task,
        eval_task=val_task,
    )
    exp.init_state()
    print_estimated_tokens(exp)
    print_chinchilla_estimate(exp.model)
    print(print_param_sizes(exp.model, depth=3)[0])
    return exp


if __name__ == "__main__":
    exp = create_experiment(
        batch_size_per_device=18,
        seq_len=2048,
        max_steps=12000,
        num_microbatches=2,
        depth=12,
        vocab_size=50257,
        checkpoint_dir="/tmp/checkpoints",
        use_multi_optimizer=True,
        muon_lr=0.0001,
        adamw_embed_lr=0.0001,
        adamw_output_lr=0.0001,
    )
    exp.outer_loop()
