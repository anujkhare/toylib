#!/usr/bin/env python3
"""
Bundled Python Project: tinystories
This file contains all project code consolidated into a single file.
"""

# ============================================================
# External Imports
# ============================================================
from jax import numpy as jnp
from transformers import AutoTokenizer
from typing import Any
import abc
import argparse
import dataclasses
import datasets as hf_datasets
import einops
import jax
import jax.numpy as jnp
import jaxtyping as jt
import math
import numpy as np
import optax
import pytest
import typing

# ============================================================
# Project Code
# ============================================================
# ============================================================
# File: toylib/nn/py
# ============================================================


def _is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


@jax.tree_util.register_pytree_node_class
class Module:
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.
    """

    def tree_flatten(self) -> tuple:
        params = []
        param_keys = []
        aux_data = dict()

        # Look through each attribute in the object
        for k, v in self.__dict__.items():
            if (
                (_is_array(v) and not _is_random_key(k))
                or isinstance(v, Module)
                or (
                    _is_supported_container(v)
                    and all(isinstance(elem, Module) for elem in v)
                )
            ):
                # trainable leaf param!
                params.append(v)
                param_keys.append(k)
            else:
                aux_data[k] = v

        aux_data["param_keys"] = param_keys
        return params, aux_data

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> "Module":
        # Create a new empty object
        obj = object.__new__(cls)

        # overwrite all of the children using the values in the given pytree
        for k, v in zip(static["param_keys"], dynamic):
            obj.__setattr__(k, v)

        for k, v in static.items():
            obj.__setattr__(k, v)

        return obj

    def __repr__(self) -> str:
        _, aux = self.tree_flatten()
        return str(aux)


# ============================================================
# File: toylib/nn/py
# ============================================================


@jax.tree_util.register_pytree_node_class
class Linear(Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    # Trainable parameters
    weights: jt.Float[jt.Array, "in_features out_features"]
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        # Split the random key for weights and bias
        w_key, b_key = jax.random.split(key, 2)

        # We initialize the weights with a uniform distribution with Xavier initialization
        # lim = 1 / math.sqrt(in_features)
        lim = math.sqrt(6 / (in_features + out_features))

        self.weights = jax.random.uniform(
            w_key, (in_features, out_features), minval=-lim, maxval=lim
        )
        self.bias = (
            jax.random.uniform(b_key, (out_features,), minval=-lim, maxval=lim)
            if use_bias
            else None
        )

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.key = key

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


@jax.tree_util.register_pytree_node_class
class Embedding(Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    # Trainable parameters
    weights: jt.Float[jt.Array, "vocab_size embedding_dim"]

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        # Initialize the embedding weights with a std normal distribution
        self.weights = jax.random.normal(key, (vocab_size, embedding_dim))

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.key = key

    def __call__(
        self, tokens: jt.Int[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0)


def rms_norm(
    x: jt.Float[jt.Array, "... dim"],
) -> jt.Float[jt.Array, "... dim"]:
    """Applies RMS Normalization over the last dimension of the input tensor.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-9)
    return x / rms


# ============================================================
# File: toylib/nn/py
# ============================================================


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
        # Use a large negative value to mask out attention logits instead of -jnp.inf
        attention_logits = jnp.where(mask, attention_logits, -1e9)

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    values = jnp.matmul(attention_weights, v)
    return values, attention_weights


@jax.tree_util.register_pytree_node_class
class MultiHeadAttention(Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    def __init__(self, qkv_dim: int, num_heads: int, *, key: jt.PRNGKeyArray) -> None:
        keys = jax.random.split(key, 4)

        # Input projections - different "heads" will be split out from the same tensor
        self.q_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[0],
        )
        self.k_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[1],
        )
        self.v_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[2],
        )

        # Output linear layer
        self.linear = Linear(in_features=qkv_dim, out_features=qkv_dim, key=keys[3])

        self.qkv_dim = qkv_dim
        self.num_heads = num_heads

    def __call__(
        self,
        Q: jt.Float[jt.Array, "... seq_len qkv_dim"],
        K: jt.Float[jt.Array, "... seq_len qkv_dim"],
        V: jt.Float[jt.Array, "... seq_len qkv_dim"],
        mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]] = None,
    ):
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)

        # Reshape the input tensors to split out the heads
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
        mask = (
            einops.rearrange(mask, "... seq_len seq_len -> ... 1 seq_len seq_len")
            if mask is not None
            else None
        )

        # Apply self atttention to each head, get the output values
        # values: [... num_heads, seq_len, qkv_dim/num_heads], attention_weights: [... num_heads, seq_len, seq_len]
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )

        values = einops.rearrange(
            values,
            "... num_heads seq_len d -> ... seq_len (num_heads d)",
        )

        # Apply linear: [..., seq_len, qkv_dim]
        values = self.linear(values)

        # return the attention weights and the output values
        return values, attention_weights


class RotaryPositionalEmbedding:
    """Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864."""

    def __init__(self, base: int = 10000):
        self.base = base

    def __call__(
        self, inputs: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        return inputs


# ============================================================
# File: py
# ============================================================


class BatchedTokenizedHFDataset:
    def __init__(
        self,
        dataset_path: str = "karpathy/fineweb-edu-100b-shuffle",
        bos_token: int = -1,
        tokenizer_name: str = "gpt2",
        split: str = "train",
        *,
        batch_size: int = 128,
        seq_len: int = 2048,
        tokenizer_batch_size: int = 8,
        streaming: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_iter = iter(
            hf_datasets.load_dataset(
                dataset_path, streaming=streaming, split=split
            ).batch(tokenizer_batch_size)
        )  # Fetch in batches

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bos_token = bos_token

        self.token_buffer = []

    def __iter__(self):
        return self

    def __next__(self) -> jnp.ndarray:
        token_needed = self.batch_size * self.seq_len + 1  # 1 for the last target token
        while len(self.token_buffer) < token_needed:
            # Load tokenizer_batch_size sequences from the dataset
            input_batch = next(self.dataset_iter)
            texts = input_batch["text"]

            # Tokenize all sequences
            tokenized = self.tokenizer(
                texts,
                return_tensors=None,  # return a list of lists
                padding=False,
                truncation=False,
                max_length=None,
            )["input_ids"]

            # Add tokens to the buffer
            for tokens in tokenized:
                self.token_buffer.append(self.bos_token)
                self.token_buffer.extend(tokens)

        # Extract needed tokens from the buffer
        tokens = self.token_buffer[:token_needed]
        self.token_buffer = self.token_buffer[token_needed:]

        # Create jax arrays for inputs and targets
        inputs = jnp.array(tokens[:-1], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        targets = jnp.array(tokens[1:], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        return {
            "inputs": inputs,
            "targets": targets,
        }


# ============================================================
# File: py
# ============================================================


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the DecoderOnlyTransformer model."""

    num_layers: int = 2
    num_heads: int = 5
    qkv_dim: int = 256

    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    seq_len: int = 512

    # logit softcap (Gemma 2: https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
    logit_softcap: float = 15.0


@jax.tree_util.register_pytree_node_class
class MLP(Module):
    def __init__(self, qkv_dim: int, *, key: jt.PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        # standard transformer architecture uses a feedforward layer
        # with an inner dimension of 4 times the model dimension.
        # See "Attention is All You Need" paper for more details.
        # https://arxiv.org/abs/1706.03762
        self.fc1 = Linear(in_features=qkv_dim, out_features=4 * qkv_dim, key=keys[0])
        self.fc2 = Linear(in_features=4 * qkv_dim, out_features=qkv_dim, key=keys[1])

    def __call__(
        self, x: jt.Float[jt.Array, "... qkv_dim"]
    ) -> jt.Float[jt.Array, "... qkv_dim"]:
        x = self.fc1(x)
        # TODO: nanochat using relu squared. why? answer: https://arxiv.org/abs/2002.05202
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


@jax.tree_util.register_pytree_node_class
class CausalSelfAttention(Module):
    def __init__(self, qkv_dim: int, num_heads: int, *, key: jt.PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.mha = MultiHeadAttention(qkv_dim=qkv_dim, num_heads=num_heads, key=keys[0])
        self.pos_emb = RotaryPositionalEmbedding()

    def _make_causal_mask(self, seq_len: int) -> jt.Float[jt.Array, "seq_len seq_len"]:
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        # TODO
        return x


@jax.tree_util.register_pytree_node_class
class DecoderBlock(Module):
    def __init__(self, qkv_dim: int, num_heads: int, *, key: jt.PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.causal_attn = CausalSelfAttention(
            qkv_dim=qkv_dim, num_heads=num_heads, key=keys[0]
        )
        self.mlp = MLP(qkv_dim=qkv_dim, key=keys[1])

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        # "Serial" implementation: y = x + MLP(LN(x + CausalSelfAttention(LN(x))))
        # A "parallel" implementation is also possible:
        #  y = x + MLP(LN(x)) + CausalSelfAttention(LN(x))
        x = x + self.causal_attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


@jax.tree_util.register_pytree_node_class
class DecoderOnlyTransformer(Module):
    """A simple decoder-only transformer model.

    Takes in a sequence of tokens, embeds them using a learned embedding layer,
    applies causal self-attention with ROPE embeddings, and outputs the logits
    for the next token prediction.
    """

    key: jt.PRNGKeyArray
    config: ModelConfig

    def __init__(self, config: ModelConfig, *, key: jt.PRNGKeyArray) -> None:
        # Generate keys - embedding, attention layers, output projection
        keys = jax.random.split(key, config.num_layers + 2)

        # Embedding layer
        self.embedding_layer = Embedding(
            vocab_size=config.vocab_size, embedding_dim=config.qkv_dim, key=keys[0]
        )

        # Self-attention layers
        self.blocks = []
        for ix in range(config.num_layers):
            self.blocks.append(
                DecoderBlock(
                    qkv_dim=config.qkv_dim,
                    num_heads=config.num_heads,
                    key=keys[ix + 1],
                )
            )

        # Output projection
        self.output_layer = Linear(
            in_features=config.qkv_dim, out_features=config.vocab_size, key=keys[-1]
        )

        self.config = config

    def __call__(
        self, x: jt.Float[jt.Array, "batch_size seq_len"]
    ) -> jt.Float[jt.Array, "batch_size seq_len vocab_size"]:
        """Forward pass for the decoder-only transformer model.

        Args:
            x: Input token ids of shape [batch_size, seq_len]

        Returns:
            Unnormalized logits over the vocabulary of shape [batch_size, seq_len, vocab_size]
        """
        # Input projection to project the embeddings to the model dimension
        x = self.embedding_layer(x)
        x = rms_norm(x)

        # Apply each attention layer
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x)

        # Output projection with softcap to prevent large logit values
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
    targets_one_hot = jax.nn.one_hot(
        targets, num_classes=logits.shape[-1]
    )  # [batch_size, seq_len, vocab_size]
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [batch_size, seq_len, vocab_size]
    per_token_loss = -jnp.sum(
        targets_one_hot * log_probs, axis=-1
    )  # [batch_size, seq_len]

    # masked loss - [batch_size, seq_len]
    masked_loss = mask * per_token_loss

    # The total loss is averaged per valid token - this treats all tokens equally
    total_loss = jnp.sum(masked_loss) / jnp.sum(mask)

    return total_loss, per_token_loss


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
    logits = model(tokens)  # doesn't use mask right now
    total_loss, _ = loss_fn(logits, targets, mask)
    return total_loss


# ============================================================
# File: py
# ============================================================

"""Basic types for the training loop and configurations."""


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 1


@dataclasses.dataclass
class Config:
    """Configuration for the experiment."""

    model_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def _serlialize_dataclass_config(config: Config) -> dict:
    result = dataclasses.asdict(config)
    for k, v in result.items():
        if dataclasses.is_dataclass(v):
            result[k] = _serlialize_dataclass_config(v)
    return result


class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config: Config, *args, **kwargs) -> None:
        self.config = config

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


class WandBLogger(Logger):
    """Logger implementation using Weights and Biases (wandb)."""

    def __init__(
        self, config: Config, project_name: str, user_name: str, *args, **kwargs
    ) -> None:
        import wandb

        self.config = config
        self.run = wandb.init(
            entity=user_name,
            project=project_name,
            config=_serlialize_dataclass_config(self.config),
        )
        self.run.define_metric("*", step_metric="global_step")

    def log(self, step: int, metrics: dict) -> None:
        metrics["global_step"] = step
        self.run.log(metrics)

    def close(self) -> None:
        self.run.finish()


class TensorBoardLogger(Logger):
    """Logger implementation that logs metrics to tensorboard locally."""

    def __init__(self, config: Config, output_path: str, *args, **kwargs) -> None:
        import os
        from tensorboardX import SummaryWriter
        import time

        self.config = config
        self.writer = SummaryWriter(
            logdir=os.path.join(output_path, time.strftime("%Y%m%d-%H%M%S"))
        )

    def log(self, step: int, metrics: dict, tag: str = "train") -> None:
        self.writer.add_scalars(tag, metrics, step)

    def close(self) -> None:
        self.writer.close()


class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(self, config: Config, output_path: str, *args, **kwargs) -> None:
        import json

        self.config = config
        self.file_ptr = open(output_path, "w")
        json.dump(_serlialize_dataclass_config(self.config), self.file_ptr, indent=4)

    def log(self, step: int, metrics: dict) -> None:
        self.temp_file.write(f"Step {step}: {metrics}\n")

    def close(self) -> None:
        self.file_ptr.close()


# ============================================================
# File: run_tokenize.py
# ============================================================

"""Script to process a text file with delimited examples, tokenize each example.

Example usage:

python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/tinystories_sample.txt \
    --output-path=toylib_projects/tinystories/data/tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

# Full TinyStories v2
python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-valid.txt \
    --output-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-valid-tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-train.txt \
    --output-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-train-tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

"""


def process_text_file(
    input_path: str,
    output_path: str,
    tokenizer: Any,
    delimiter: str = "\n",
    encoding: str = "utf-8",
    max_sequence_length: int = 1024,
) -> None:
    """
    Process a text file with delimited examples, tokenize each example,
    and save as a numpy array.

    Args:
        input_path: path to input text file
        output_path: path to save the numpy array (.npy file)
        tokenizer_version: Version/type of tokenizer to use
        delimiter: Delimiter separating examples (default: newline)
        encoding: Text file encoding (default: utf-8)
    """
    # Read the text file
    with open(input_path, "r", encoding=encoding) as f:
        content = f.read()

    # Split into examples
    examples = content.split(delimiter)

    # Filter out empty examples
    examples = [ex.strip() for ex in examples if ex.strip()]

    # Tokenize each example
    tokenized_examples = [
        np.array(
            tokenizer(example, truncation=True, max_length=max_sequence_length)[
                "input_ids"
            ],
            dtype=np.uint16,
        )
        for example in examples
    ]

    # Concatenate all tokenized examples into a single array
    token_array = np.array(tokenized_examples, dtype=object)

    # Save to disk
    np.save(output_path, token_array, allow_pickle=True)
    print(f"Saved {token_array.shape[0]} tokenized examples to {output_path}")

    # Calculate the lengths of each example and save
    lengths = [len(ex) for ex in tokenized_examples]
    np.save(
        output_path.replace(".npy", "_lengths.npy"), np.array(lengths, dtype=np.uint16)
    )
    print(
        f"Example lengths: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) / len(lengths):.2f}"
    )


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Process text file with delimited examples and tokenize them"
    )

    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to input text file"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the tokenized numpy array (.npy file)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="gpt2",
        help="Tokenizer version from transformers to use (e.g., 'gpt2')",
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        required=False,
        default=1024,
        help="Maximum sequence length for tokenization (default: 1024)",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n",
        help="Delimiter separating examples (default: newline)",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text file encoding (default: utf-8)",
    )

    return parser.parse_args()


# ============================================================
# File: test_data.py
# ============================================================

"""Tests for py."""


class TestHFDataset:
    def test_smoke(self):
        """Test that we can fetch data from the internet! Non-hermetic: requires internet access."""
        dataset = BatchedTokenizedHFDataset(
            bos_token=10000,
            batch_size=4,
            seq_len=1024,
            tokenizer_batch_size=2,
        )

        # Fetch a batch
        batch = next(dataset)
        inputs = batch["inputs"]
        targets = batch["targets"]

        assert inputs.shape == (4, 1024)
        assert targets.shape == (4, 1024)
        assert (inputs[:, 1:] == targets[:, :-1]).all().tolist()


# ============================================================
# File: test_model.py
# ============================================================

"""Tests for model.py."""


class TestDecoderOnlyTransformer:
    @pytest.mark.parametrize(
        "input_tokens,model_config",
        [
            (
                jnp.array([[1, 2, 3], [4, 5, 6]]),
                ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_smoke(self, input_tokens: jnp.ndarray, model_config: ModelConfig):
        """Test a simple forward pass of the model."""
        model = DecoderOnlyTransformer(config=model_config, key=jax.random.PRNGKey(42))
        actual = model(input_tokens)
        assert actual.shape == (
            input_tokens.shape[0],
            input_tokens.shape[1],
            model_config.vocab_size,
        )


class TestTrainStep:
    @pytest.mark.parametrize(
        "input_tokens,target_tokens,model_config",
        [
            (
                jnp.array([[1, 2, 3], [4, 5, 6]]),
                jnp.array([[2, 3, 4], [5, 6, 7]]),
                ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_smoke(
        self,
        input_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        model_config: ModelConfig,
    ):
        """Test a simple training step."""
        model = DecoderOnlyTransformer(config=model_config, key=jax.random.PRNGKey(42))
        loss = train_step(
            model, input_tokens, target_tokens, jnp.ones_like(target_tokens)
        )

        assert loss >= 0.0


# ============================================================
# File: train.py
# ============================================================


# import local modules


def main():
    config = Config(
        model_config=ModelConfig(
            vocab_size=50257,  # GPT-2 tokenizer vocab size
        ),
        training_config=TrainingConfig(),
    )

    # Dataloader
    dataset = BatchedTokenizedHFDataset(
        bos_token=1000, batch_size=128, seq_len=512, tokenizer_batch_size=8
    )

    # Model
    model = DecoderOnlyTransformer(
        config=config.model_config, key=jax.random.PRNGKey(0)
    )

    # Logger
    logger = TensorBoardLogger(config, output_path="./tensorboard_logs")

    # Optimizer
    optimizer = optax.adam(learning_rate=config.training_config.learning_rate)

    def log_metrics(logger: Logger, step: int, loss_val: float, updates):
        leaves, _ = jax.tree_util.tree_flatten(updates)
        metrics = {
            "train/loss": float(loss_val),
            "train/learning_rate": config.training_config.learning_rate,
            "gradients/0/mean": leaves[0].mean(),
            "gradients/1/mean": leaves[1].mean(),
            "gradients/2/mean": leaves[2].mean(),
        }
        logger.log(step=step, metrics=metrics)

    # Optimizer
    opt_state = optimizer.init(model)

    # Value and gradient
    loss_and_grad_fn = jax.jit(jax.value_and_grad(train_step))

    step = 0

    # Training loop
    for epoch in range(config.training_config.num_epochs):
        for batch in dataset:
            inputs, targets = batch["inputs"], batch["targets"]
            mask = jax.numpy.ones_like(inputs)

            # Compute loss and gradients
            loss_val, grads = loss_and_grad_fn(model, inputs, mask, targets)

            # Apply gradients
            updates, opt_state = optimizer.update(grads, opt_state)
            model = optax.apply_updates(model, updates)

            # Log metrics
            log_metrics(logger, step, loss_val, updates)

            # Increment step
            step += 1
