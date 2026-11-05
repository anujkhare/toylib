"""The CORE metric from the DCLM paper (https://arxiv.org/abs/2406.11794), as nanochat runs it.

22 in-context tasks, each scored by accuracy and centered on its random baseline, so 0 is
chance and 1 is perfect; CORE is the mean of the centered scores. Prompts, few-shot
examples and scoring follow nanochat's `core_eval.py`, so the numbers compare with its
reports:

- Few-shot examples for item i are drawn with `random.Random(1234 + i)`, never item i itself.
- Every prompt starts with a document separator: our `<|endoftext|>`, nanochat's BOS.
- Multiple choice and schema tasks pick the option with the lowest mean loss over the
  tokens that differ between options. Language modeling tasks count an item correct only
  if greedy decoding reproduces every token of the continuation.
- A prompt longer than the model's context keeps its last tokens.

Unlike nanochat, which runs one item at a time, all prompts of a task are sorted by length
and packed into batches of a fixed token budget, which is much faster on a GPU.

    # The d20 run, from a local or gs:// checkpoint directory
    python -m toylib_projects.p02_gpt_web.scripts.core_eval --run d20_nanochat \\
        --checkpoint-dir gs://tinystories-checkpoints/toylib-p02-climbmix/d20_nanochat

    # Hugging Face GPT-2, to check this script against the bundle's reference scores
    # (needs torch)
    python -m toylib_projects.p02_gpt_web.scripts.core_eval --hf gpt2

The eval bundle is nanochat's, downloaded once to data/eval_bundle. Besides the tasks it
holds GPT-2's reference scores, `openai-community-gpt2*.csv`.
"""

import argparse
import csv
import dataclasses
import json
import pathlib
import random
import time
import typing
import zipfile

import jinja2
import numpy as np
import yaml

from toylib.data import download

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR = pathlib.Path(__file__).parents[1] / "data" / "eval_bundle"

# forward(tokens [batch, time]) -> (loss of predicting each next token, argmax prediction),
# both [batch, time]. The last column's values are never read.
Forward = typing.Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass(frozen=True)
class Task:
    label: str
    task_type: str  # multiple_choice, schema or language_modeling
    num_fewshot: int
    continuation_delimiter: str
    random_baseline: float  # In percent, from eval_meta_data.csv.
    data: list[dict]


@dataclasses.dataclass(frozen=True)
class Prompt:
    """One sequence to score: the tokens [start, end) are the ones judged."""

    item: int
    option: int
    tokens: list[int]
    start: int
    end: int


# -----------------------------------------------------------------------------
# The eval bundle


def ensure_bundle(bundle_dir: pathlib.Path = EVAL_BUNDLE_DIR) -> pathlib.Path:
    if not (bundle_dir / "core.yaml").exists():
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        archive = download.fetch(EVAL_BUNDLE_URL, bundle_dir.parent / "eval_bundle.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(bundle_dir.parent)
    return bundle_dir


def load_tasks(bundle_dir: pathlib.Path, max_per_task: int = -1) -> list[Task]:
    with open(bundle_dir / "eval_meta_data.csv", encoding="utf-8") as f:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) for row in csv.DictReader(f)}
    with open(bundle_dir / "core.yaml", encoding="utf-8") as f:
        configs = yaml.safe_load(f)["icl_tasks"]
    tasks = []
    for config in configs:
        with open(bundle_dir / "eval_data" / config["dataset_uri"], encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        # The same shuffle as nanochat, so a subset is the same subset.
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        tasks.append(
            Task(
                label=config["label"],
                task_type=config["icl_task_type"],
                num_fewshot=config["num_fewshot"][0],
                continuation_delimiter=config.get("continuation_delimiter", " "),
                random_baseline=baselines[config["label"]],
                data=data,
            )
        )
    return tasks


def read_csv(path: pathlib.Path) -> dict[str, tuple[float | None, float]]:
    """(accuracy, centered) by task from a CSV in the bundle's format; "CORE" has no accuracy."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in list(f)[1:]:
            label, accuracy, score = (cell.strip() for cell in line.split(","))
            out[label] = (float(accuracy) if accuracy else None, float(score))
    return out


def write_csv(path: pathlib.Path, out: dict) -> None:
    """In the bundle's reference format, so the files compare line by line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label, accuracy in out["results"].items():
            f.write(f"{label:<35}, {accuracy:<10.6f}, {out['centered_results'][label]:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {out['core_metric']:<10.6f}\n")


# -----------------------------------------------------------------------------
# Prompts, as nanochat renders them

_MC = jinja2.Template(
    """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
)
_SCHEMA = jinja2.Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
)
_LM = jinja2.Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
)


def common_length(sequences: list[list[int]], from_end: bool = False) -> int:
    """How many leading (or trailing) tokens all sequences share."""
    n = min(len(s) for s in sequences)
    for i in range(n):
        ix = -1 - i if from_end else i
        if any(s[ix] != sequences[0][ix] for s in sequences):
            return i
    return n


def task_prompts(
    task: Task, encode: typing.Callable[[str], list[int]], bos: int, max_len: int
) -> list[Prompt]:
    """Every sequence a task needs scored, cropped to `max_len` from the left."""
    prompts = []
    for idx, item in enumerate(task.data):
        fewshot = []
        if task.num_fewshot > 0:
            others = [i for i in range(len(task.data)) if i != idx]
            # Fewer only when --max-per-task leaves too few items; full runs match nanochat.
            k = min(task.num_fewshot, len(others))
            fewshot = [task.data[i] for i in random.Random(1234 + idx).sample(others, k)]
        context = dict(
            fewshot_examples=fewshot, continuation_delimiter=task.continuation_delimiter, item=item
        )

        if task.task_type == "multiple_choice":
            texts = [_MC.render(choice=choice, **context) for choice in item["choices"]]
            tokens = [[bos] + encode(t) for t in texts]
            start = common_length(tokens)
            spans = [(start, len(t)) for t in tokens]
        elif task.task_type == "schema":
            texts = [_SCHEMA.render(context=option, **context) for option in item["context_options"]]
            tokens = [[bos] + encode(t) for t in texts]
            suffix = common_length(tokens, from_end=True)
            spans = [(len(t) - suffix, len(t)) for t in tokens]
        elif task.task_type == "language_modeling":
            without = [bos] + encode(_LM.render(include_continuation=False, **context).strip())
            with_ = [bos] + encode(_LM.render(include_continuation=True, **context))
            if with_[: len(without)] != without or len(with_) <= len(without):
                raise ValueError(f"{task.label} item {idx}: the context doesn't tokenize as a prefix")
            tokens, spans = [with_], [(len(without), len(with_))]
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")

        for option, (t, (start, end)) in enumerate(zip(tokens, spans)):
            crop = max(0, len(t) - max_len)
            prompts.append(Prompt(idx, option, t[crop:], start - crop, end - crop))
    return prompts


# -----------------------------------------------------------------------------
# Scoring


def bucket(length: int, smallest: int = 64) -> int:
    """The next power of two, so a jitted forward compiles for few shapes."""
    return max(smallest, 1 << (length - 1).bit_length())


def score_prompts(
    forward: Forward, prompts: list[Prompt], pad: int, token_budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean loss over each prompt's judged tokens, and whether greedy decoding matches them all."""
    mean_loss = np.zeros(len(prompts))
    all_match = np.zeros(len(prompts), dtype=bool)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].tokens))
    i = 0
    while i < len(order):
        width = bucket(len(prompts[order[i]].tokens))
        # A batch fills the token budget with prompts that fit its width. Its row count
        # depends only on the width, so the forward compiles once per width.
        rows = max(1, token_budget // width)
        batch = [j for j in order[i : i + rows] if len(prompts[j].tokens) <= width]
        tokens = np.full((rows, width), pad, dtype=np.int32)
        for r, j in enumerate(batch):
            tokens[r, : len(prompts[j].tokens)] = prompts[j].tokens
        losses, predictions = forward(tokens)
        for r, j in enumerate(batch):
            p = prompts[j]
            # Position t predicts token t + 1.
            mean_loss[j] = losses[r, p.start - 1 : p.end - 1].mean()
            all_match[j] = np.array_equal(
                predictions[r, p.start - 1 : p.end - 1], tokens[r, p.start : p.end]
            )
        i += len(batch)
    return mean_loss, all_match


def task_accuracy(
    task: Task, prompts: list[Prompt], mean_loss: np.ndarray, all_match: np.ndarray
) -> float:
    correct = np.zeros(len(task.data), dtype=bool)
    if task.task_type == "language_modeling":
        for p, match in zip(prompts, all_match):
            correct[p.item] = match
    else:
        best: dict[int, tuple[float, int]] = {}
        for p, loss in zip(prompts, mean_loss):
            # Ties go to the first option, as with nanochat's list.index(min(...)).
            if p.item not in best or loss < best[p.item][0]:
                best[p.item] = (loss, p.option)
        for item, (_, option) in best.items():
            correct[item] = option == task.data[item]["gold"]
    return float(correct.mean())


def centered(accuracy: float, random_baseline: float) -> float:
    return (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)


def evaluate(
    forward: Forward,
    encode: typing.Callable[[str], list[int]],
    bos: int,
    max_len: int,
    tasks: list[Task],
    token_budget: int = 16_384,
) -> dict:
    results, centered_results = {}, {}
    for task in tasks:
        t0 = time.time()
        prompts = task_prompts(task, encode, bos, max_len)
        mean_loss, all_match = score_prompts(forward, prompts, bos, token_budget)
        accuracy = task_accuracy(task, prompts, mean_loss, all_match)
        results[task.label] = accuracy
        centered_results[task.label] = centered(accuracy, task.random_baseline)
        print(
            f"{task.label:34s} {task.num_fewshot:2d}-shot n={len(task.data):6d} "
            f"accuracy {accuracy:.4f} centered {centered_results[task.label]:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    core = float(np.mean(list(centered_results.values())))
    print(f"CORE {core:.4f}", flush=True)
    return {"results": results, "centered_results": centered_results, "core_metric": core}


# -----------------------------------------------------------------------------
# Models


def jax_forward(model) -> Forward:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def run(model, tokens):
        logits = model(tokens).astype(jnp.float32)
        targets = jnp.roll(tokens, -1, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        return losses, logits.argmax(-1)

    return lambda tokens: tuple(np.asarray(x) for x in run(model, tokens))


def load_run(run_name: str, checkpoint_dir: str, step: int | None, attention: str | None):
    """Our model and tokenizer: (forward, encode, bos, max_len)."""
    import jax

    from toylib.harness import inference
    from toylib_projects.p02_gpt_web import experiment, tokenize_webtext, train

    run = dataclasses.replace(train.RUNS[run_name], attention_impl=attention)
    template = experiment.build_model(train.model_config(run), jax.random.key(0))
    model = inference.load_model(template, checkpoint_dir, step)
    tokenizer = tokenize_webtext.load_tokenizer(
        run.vocab_size, tokenize_webtext.data_dir_for(run.num_train_shards)
    )
    return jax_forward(model), tokenizer.encode, tokenize_webtext.EOT_ID, run.seq_len


def load_hf(name: str):
    """A Hugging Face causal LM, as nanochat evaluates them: (forward, encode, bos, max_len)."""
    import torch
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def forward(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(tokens).long().to(device)
        logits = model(x).logits.float()
        targets = torch.roll(x, -1, dims=1)
        losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).view(x.shape)
        return losses.cpu().numpy(), logits.argmax(-1).cpu().numpy()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return forward, encode, tokenizer.bos_token_id, model.config.max_position_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", help="A key of train.RUNS, e.g. d20_nanochat")
    parser.add_argument("--checkpoint-dir", help="Local or gs:// directory of the run's checkpoints")
    parser.add_argument("--step", type=int, default=None, help="Default: the latest")
    parser.add_argument("--attention", default=None, help="attention_impl: xla or cudnn. Default: masked matmuls")
    parser.add_argument("--hf", help="A Hugging Face model instead, e.g. gpt2")
    parser.add_argument("--max-per-task", type=int, default=-1, help="-1 runs every item")
    parser.add_argument("--token-budget", type=int, default=16_384, help="Tokens per forward pass")
    parser.add_argument("--output", type=pathlib.Path, help="CSV to write, in the bundle's format")
    args = parser.parse_args()
    if bool(args.run) == bool(args.hf):
        parser.error("Pass exactly one of --run and --hf")
    if args.run and not args.checkpoint_dir:
        parser.error("--run needs --checkpoint-dir")

    tasks = load_tasks(ensure_bundle(), args.max_per_task)
    if args.hf:
        forward, encode, bos, max_len = load_hf(args.hf)
    else:
        forward, encode, bos, max_len = load_run(args.run, args.checkpoint_dir, args.step, args.attention)
    out = evaluate(forward, encode, bos, max_len, tasks, args.token_budget)
    if args.output:
        write_csv(args.output, out)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
