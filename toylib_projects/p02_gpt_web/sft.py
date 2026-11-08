"""Supervised fine-tuning: a pretrained run learns to chat, as in nanochat's `chat_sft.py`.

nanochat used to fine-tune in two stages, midtraining then SFT; since January 2026 it is
one, and this follows that one. Against pretraining:

- Data: one pass over nanochat's mixture of conversations (`sft_data.py`), in its chat
  format, with the loss on the assistant's tokens only.
- Tokens: our tokenizer has no chat tokens, so the eight nanochat adds are new ids, and
  the embeddings, value embeddings and output layer each grow a row for them. The new
  rows start as the model would initialize them; Adam's moments for them start at zero.
- Optimizer: the same six groups and peak rates, warm-started with pretraining's
  momentum and Adam moments. Every rate starts at 80% of its peak (pretraining ended at
  5%), holds for half the run, then falls linearly to zero. Muon's momentum ramps from
  0.85 to 0.95 over 300 steps, and its weight decay, which pretraining ended at zero,
  stays there. The AdamW groups keep their small weight decays.
- Evaluation: bits per byte on the validation mixture's assistant tokens, every 200
  steps, and ChatCORE: every multiple-choice problem but 24 of each generative task. At
  the end, ChatCORE on every problem.

The optimizer's step count carries on from pretraining, which Adam's bias correction
needs, so the schedules here count from where pretraining stopped.

    uv run python -m toylib_projects.p02_gpt_web.sft_data   # Once: render and pack.
    uv run python -m toylib_projects.p02_gpt_web.sft --run d20_sft \\
        --checkpoint-dir gs://tinystories-checkpoints/toylib-p02-climbmix/d20_sft
"""

import argparse
import dataclasses
import json
import pathlib
import time
import typing

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from toylib.harness import inference
from toylib.harness import metrics as harness_metrics
from toylib.nn import module
from toylib_projects.p02_gpt_web import chat_format
from toylib_projects.p02_gpt_web import experiment
from toylib_projects.p02_gpt_web import generate
from toylib_projects.p02_gpt_web import metrics as metrics_module
from toylib_projects.p02_gpt_web import sft_data
from toylib_projects.p02_gpt_web import tokenize_webtext
from toylib_projects.p02_gpt_web import train
from toylib_projects.p02_gpt_web.scripts import chat_eval

OUTPUT_DIR = train.OUTPUT_DIR
CHAT_PROMPTS = [
    "What is the capital of France?",
    "Why is the sky blue?",
    "Write a haiku about the ocean.",
    "If 5*x + 3 = 13, what is x?",
    "Who wrote Pride and Prejudice?",
    "Give me three tips for sleeping better.",
]


@dataclasses.dataclass(frozen=True)
class SFTRun:
    """Fine-tuning settings; batch shape, architecture and dtype come from the base run."""

    base: str | train.Run  # A key of train.RUNS.
    base_checkpoint_dir: str = ""
    base_step: int | None = None  # Default: the latest.
    max_steps: int | None = None  # Default: one pass over the training rows.
    init_lr_frac: float = 0.8
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    momentum_warmup_steps: int = 300
    eval_interval_steps: int = 200
    eval_tokens: int = 40 * 524_288
    chatcore_interval_steps: int = 200  # 0 turns it off.
    chatcore_max_categorical: int | None = None  # Default: every problem.
    chatcore_max_generative: int = 24
    save_interval_steps: int = 150
    attention_impl: str | None = "base"  # "base" keeps the base run's.

    @property
    def base_run(self) -> train.Run:
        return train.RUNS[self.base] if isinstance(self.base, str) else self.base


D20_CHECKPOINTS = "gs://tinystories-checkpoints/toylib-p02-climbmix/d20_nanochat"
RUNS = {"d20_sft": SFTRun(base="d20_nanochat", base_checkpoint_dir=D20_CHECKPOINTS)}


def model_run(run: SFTRun) -> train.Run:
    """The base run's model, with a row for each chat token."""
    base = run.base_run
    vocab_size = base.vocab_size + len(chat_format.SPECIAL_TOKENS)
    impl = base.attention_impl if run.attention_impl == "base" else run.attention_impl
    return dataclasses.replace(base, vocab_size=vocab_size, attention_impl=impl)


def load_tokenizer(run: SFTRun) -> chat_format.ChatTokenizer:
    base = run.base_run
    data_dir = tokenize_webtext.data_dir_for(base.num_train_shards)
    return chat_format.ChatTokenizer(tokenize_webtext.load_tokenizer(base.vocab_size, data_dir))


# --- Schedules ------------------------------------------------------------------------


def lr_multiplier(
    max_steps: int, init_lr_frac: float = 0.8, warmdown_ratio: float = 0.5,
    final_lr_frac: float = 0.0, offset: int = 0,
) -> optax.Schedule:
    """nanochat's SFT `get_lr_multiplier` with no warmup, by step `count - offset`."""

    def schedule(count):
        progress = (jnp.asarray(count, jnp.float32) - offset) / max_steps
        decay = (progress - (1 - warmdown_ratio)) / warmdown_ratio
        warmdown = (1 - decay) + decay * final_lr_frac
        return init_lr_frac * jnp.where(progress <= 1 - warmdown_ratio, 1.0, warmdown)

    return schedule


def muon_momentum(warmup_steps: int = 300, offset: int = 0) -> optax.Schedule:
    """nanochat's SFT `get_muon_momentum`: 0.85 to 0.95 over `warmup_steps`."""

    def schedule(count):
        frac = jnp.clip((jnp.asarray(count, jnp.float32) - offset) / warmup_steps, 0.0, 1.0)
        return 0.85 + 0.10 * frac

    return schedule


# --- The experiment -------------------------------------------------------------------


@dataclasses.dataclass
class SFTExperiment(experiment.Experiment):
    """Adds chat samples and ChatCORE to the evaluation."""

    chatcore_interval_steps: int = 0
    chatcore_limits: dict = dataclasses.field(default_factory=dict)
    chatcore_max_new_tokens: int = 512
    chat_prompts: list[str] = dataclasses.field(default_factory=lambda: list(CHAT_PROMPTS))

    @property
    def chat_tokenizer(self) -> chat_format.ChatTokenizer:
        return self.train_task.dataset.tokenizer

    def eval_model(self):
        """The model with masked attention, which runs any shape; cuDNN's is for training."""
        template = getattr(self, "_eval_template", None)
        if template is None:
            config = dataclasses.replace(self.model_config, attention_impl=None)
            self._eval_template = template = jax.eval_shape(
                lambda: experiment.build_model(config, jax.random.key(0))
            )
        return jax.tree.unflatten(jax.tree.structure(template), jax.tree.leaves(self.model))

    def sampling_evaluation(self) -> None:
        tokenizer = self.chat_tokenizer
        prompts = [
            chat_format.render_for_completion(tokenizer, [{"role": "user", "content": p}])
            for p in self.chat_prompts
        ]
        outs = generate.greedy_batch(
            self.eval_model(), prompts, self.sampling_max_tokens,
            stop_tokens={tokenizer.assistant_end, chat_format.BOS},
            tool=lambda ids: chat_format.tool_tokens(tokenizer, ids),
        )
        samples = [{"prompt": p, "output": tokenizer.decode(o)} for p, o in zip(self.chat_prompts, outs)]
        self.logger_obj.log(self.step, metrics={"step": self.step, "samples": samples})

    def chatcore(self, limits: dict | None = None, verbose: bool = False) -> tuple[dict, list[dict]]:
        """ChatCORE on the current weights, logged as chatcore/... at this step."""
        start = time.perf_counter()
        max_new = self.chatcore_max_new_tokens
        last_logits, complete, fmt = chat_eval.model_functions(self.eval_model(), self.chat_tokenizer, max_new)
        scores, completions = chat_eval.evaluate(
            last_logits, complete, fmt, self.model_config.seq_len,
            max_problems=self.chatcore_limits if limits is None else limits,
            max_new_tokens=max_new, verbose=verbose,
        )
        logs = {f"chatcore/{task}": acc for task, acc in scores["accuracy"].items()}
        logs.update({f"chatcore/{k}": scores[k] for k in ("ChatCORE", "ChatCORE_cat") if k in scores})
        logs["chatcore/seconds"] = time.perf_counter() - start
        self.logger_obj.log(self.step, metrics=logs)
        print(f"step {self.step} ChatCORE {json.dumps(logs)}", flush=True)
        return scores, completions

    def eval(self):
        super().eval()
        if self.chatcore_interval_steps and self.step % self.chatcore_interval_steps == 0:
            self.chatcore()


def create_experiment(
    run: SFTRun,
    run_id: str = "sft",
    data_dir: pathlib.Path = sft_data.DATA_DIR,
    output_dir: pathlib.Path = OUTPUT_DIR,
    checkpoint_dir: str | None = None,
    logger_config: experiment.LoggerConfig | None = None,
    tokenizer: chat_format.ChatTokenizer | None = None,
) -> SFTExperiment:
    """The fine-tuning experiment, initialized but not yet loaded with the base weights."""
    base, mrun = run.base_run, model_run(run)
    tokenizer = tokenizer or load_tokenizer(run)
    if len(tokenizer) != mrun.vocab_size:
        raise ValueError(f"The tokenizer has {len(tokenizer)} ids, the model {mrun.vocab_size}")
    run_dir = pathlib.Path(output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bpt_path = run_dir / "bytes_per_token.npy"
    np.save(bpt_path, tokenizer.bytes_per_token())

    microbatch = base.batch_size // base.num_microbatches
    train_paths = sft_data.paths("train", mrun.vocab_size, data_dir)
    val_paths = sft_data.paths("val", mrun.vocab_size, data_dir)
    train_rows = sft_data.PackedRowsDataset(*map(str, train_paths), batch_size=base.batch_size, tokenizer=tokenizer)
    val_rows = sft_data.PackedRowsDataset(*map(str, val_paths), batch_size=microbatch, repeatable=True)
    max_steps = run.max_steps or train_rows.steps_per_epoch

    # The optimizer's count continues from pretraining's last step.
    offset = base.max_steps
    config = train.model_config(mrun)
    tokens_per_step = base.batch_size * base.seq_len
    schedule = dict(max_steps=max_steps, init_lr_frac=run.init_lr_frac,
                    warmdown_ratio=run.warmdown_ratio, final_lr_frac=run.final_lr_frac)
    optimizer_config = train.create_nanochat_optimizer_config(
        config,
        max_steps=max_steps,
        tokens_per_step=tokens_per_step,
        lr_multiplier=lr_multiplier(**schedule, offset=offset),
        muon_momentum=muon_momentum(run.momentum_warmup_steps, offset=offset),
        muon_weight_decay=lambda count: jnp.zeros([]),
    )
    limits = {t: run.chatcore_max_categorical for t in chat_eval.CATEGORICAL}
    limits.update({t: run.chatcore_max_generative for t in chat_eval.TASKS if t not in chat_eval.CATEGORICAL})

    exp = SFTExperiment(
        model_config=config,
        training_config=experiment.TrainingConfig(
            max_steps=max_steps,
            num_microbatches=base.num_microbatches,
            max_grad_norm=base.max_grad_norm,
            optimizer_config=optimizer_config,
        ),
        eval_config=experiment.EvalConfig(
            eval_interval_steps=run.eval_interval_steps,
            num_eval_steps=max(1, run.eval_tokens // (microbatch * base.seq_len)),
        ),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=run.save_interval_steps,
            max_to_keep=2,
            checkpoint_dir=checkpoint_dir or str(run_dir / "checkpoints"),
            checkpoint_dataset_iterator=True,
        ),
        logger_config=logger_config
        or experiment.LoggerConfig(log_dir=str(run_dir), run_id=run_id, train_log_interval_steps=10),
        train_task=experiment.Task(name="train", dataset=train_rows),
        eval_task=experiment.Task(
            name="val",
            dataset=val_rows,
            metrics=[harness_metrics.Loss(), metrics_module.BitsPerByte(str(bpt_path))],
        ),
        sampling_max_tokens=128,
        tokens_per_step=tokens_per_step,
        flops_per_token=metrics_module.flops_per_token(config),
        lr_multiplier=lr_multiplier(**schedule),
        chatcore_interval_steps=run.chatcore_interval_steps,
        chatcore_limits=limits,
    )
    exp.init_state()
    print(f"Fine-tuning for {max_steps} steps of {base.batch_size} rows "
          f"({train_rows.steps_per_epoch} make one pass)", flush=True)
    return exp


def optimizer_counts(opt_state) -> set[int]:
    """Every step count in an optimizer state."""
    flat = jax.tree_util.tree_flatten_with_path(opt_state)[0]
    return {int(v) for path, v in flat if getattr(path[-1], "name", None) == "count"}


def load_base(exp: SFTExperiment, run: SFTRun) -> None:
    """Load the pretrained weights and optimizer state, growing them for the chat tokens."""
    manager = ocp.CheckpointManager(
        run.base_checkpoint_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    )
    step = run.base_step if run.base_step is not None else manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoints in {run.base_checkpoint_dir!r}")
    model_template = exp._unreplicate_for_checkpoint(exp.model)
    opt_template = exp._unreplicate_for_checkpoint(exp.opt_state)
    frozen = jax.tree.map(lambda m: not m, module.trainable_mask(model_template))
    model = inference.restore_tree(manager, step, "model", model_template, frozen, grow=True)
    opt_state = inference.restore_tree(manager, step, "opt_state", opt_template, grow=True)
    counts = optimizer_counts(opt_state)
    if counts != {run.base_run.max_steps}:
        raise ValueError(f"Expected the base run's optimizer at step {run.base_run.max_steps}, not {counts}")
    exp.model = jax.device_put(model, exp.replicated_sharding)
    exp.opt_state = jax.device_put(opt_state, exp.replicated_sharding)
    manager.close()
    print(f"Loaded {run.base_checkpoint_dir} at step {step}", flush=True)


def finish(exp: SFTExperiment, output_dir: pathlib.Path) -> dict:
    """Full validation and ChatCORE on every problem; writes chatcore.json and completions."""
    val = exp.run_validation()
    scores, completions = exp.chatcore(limits={}, verbose=True)
    scores["val"] = val
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chatcore.json").write_text(json.dumps(scores, indent=2))
    (output_dir / "chatcore_completions.jsonl").write_text("".join(json.dumps(c) + "\n" for c in completions))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=sorted(RUNS), default="d20_sft")
    parser.add_argument("--run-id", help="Output folder name (default: the --run name)")
    parser.add_argument("--data-dir", type=pathlib.Path, default=sft_data.DATA_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", help="Can be gs://. Default: under outputs/")
    parser.add_argument("--base-checkpoint-dir", help="Default: the run's")
    parser.add_argument("--max-steps", type=int, help="Default: one pass over the data")
    parser.add_argument("--attention", choices=["xla", "cudnn"], help="Default: the base run's")
    parser.add_argument("--wandb-project", help="Default: log to a file only")
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()

    run, run_id = RUNS[args.run], args.run_id or args.run
    overrides = {}
    if args.base_checkpoint_dir:
        overrides["base_checkpoint_dir"] = args.base_checkpoint_dir
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.attention:
        overrides["attention_impl"] = None if args.attention == "xla" else args.attention
    run = dataclasses.replace(run, **overrides)
    logger_config = None
    if args.wandb_project:
        logger_config = experiment.WandBLoggerConfig(
            project_name=args.wandb_project, user_name=args.wandb_entity,
            log_dir=str(args.output_dir / run_id), run_id=run_id,
            train_log_interval_steps=10, also_log_to_file=True,
        )
    exp = create_experiment(run, run_id, args.data_dir, args.output_dir, args.checkpoint_dir, logger_config)
    if not exp.resume():
        load_base(exp, run)
    exp.outer_loop()
    scores = finish(exp, args.output_dir / run_id)
    print("FINAL", json.dumps({k: scores.get(k) for k in ("ChatCORE", "ChatCORE_cat", "accuracy", "val")}), flush=True)
    exp.cleanup()


if __name__ == "__main__":
    main()
