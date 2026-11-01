"""The shared training harness, set up for the decoder-only language model.

``toylib.harness.experiment`` does the training. This module adds what is
specific to language modelling: the default model factory and loss functions,
and a sampling evaluation that decodes generated text with the tokenizer.

The harness's config classes are re-exported, so LM scripts configure a run
from this one module (``experiment.TrainingConfig``, ``experiment.Task``, ...).
"""

import dataclasses
import typing

import jax  # noqa: F401  (tests patch experiment.jax)
import jax.numpy as jnp
import optax  # noqa: F401  (tests patch experiment.optax)

from toylib.harness import experiment as harness
from toylib.harness.experiment import (  # noqa: F401  (re-exported)
    CheckpointConfig,
    EvalConfig,
    LoggerConfig,
    MultiOptimizerConfig,
    OptimizerConfig,
    Task,
    TrainingConfig,
    WandBLoggerConfig,
    _serialize_dataclass_config,
)
from toylib_projects.tinystories import decoder_only_model


DEFAULT_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


def build_model(
    config: decoder_only_model.ModelConfig, key: jax.Array
) -> decoder_only_model.DecoderOnlyTransformer:
    """Construct and initialize the transformer (the harness's ``model_factory``)."""
    model = decoder_only_model.DecoderOnlyTransformer(config=config, key=key)
    model.init()
    return model


def train_forward(model, batch):
    """Training loss; no per-token outputs, to keep the microbatch scan light."""
    return decoder_only_model.train_step(model, batch, return_aux=False)


def eval_forward(model, batch):
    """Eval loss plus per-token losses, which ``BitsPerByte`` needs."""
    return decoder_only_model.train_step(model, batch, return_aux=True)


@dataclasses.dataclass
class Experiment(harness.Experiment):
    """``harness.Experiment`` with language-model defaults and text sampling."""

    forward_fn: typing.Callable[..., typing.Any] = train_forward
    model_factory: typing.Callable[..., typing.Any] = build_model
    eval_forward_fn: typing.Callable[..., typing.Any] | None = eval_forward
    model_config: decoder_only_model.ModelConfig = dataclasses.field(
        default_factory=decoder_only_model.ModelConfig
    )

    def sampling_evaluation(
        self, prompts: list[str] | None = None, max_tokens: int = 10
    ) -> None:
        """Sample a continuation of each prompt and log the decoded text.

        Runs on a single device for simplicity. Skipped when the dataset has no
        tokenizer (e.g. the dummy dataset used for compilation analysis).
        """
        self._assert_initialized()
        tokenizer = getattr(self.train_task.dataset, "tokenizer", None)
        if tokenizer is None:
            return
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        tokenized_prompts = tokenizer(
            prompts,
            return_tensors=None,
            padding=False,
            truncation=False,
            max_length=None,
        )["input_ids"]

        seq_len = self.model_config.seq_len
        results = []
        for ix, prompt_tokens in enumerate(tokenized_prompts):
            padded = jnp.zeros(seq_len, dtype=jnp.uint16)
            padded = padded.at[: len(prompt_tokens)].set(
                jnp.array(prompt_tokens, dtype=jnp.uint16)
            )
            generated = decoder_only_model.sample(
                model=self.model,
                input_tokens=padded,
                prompt_len=len(prompt_tokens),
                key=jax.random.key(0),
                max_output_tokens=max_tokens,
                temperature=1.0,
                top_k=5,
            )
            results.append(
                {"prompt": prompts[ix], "output": tokenizer.decode(generated.tolist())}
            )
        self.logger_obj.log(self.step, metrics={"step": self.step, "samples": results})
