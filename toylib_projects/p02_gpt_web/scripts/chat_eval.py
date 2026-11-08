"""ChatCORE, nanochat's benchmark for its chat models, on one of our runs.

Five tasks, each centered on its random baseline (0 is chance, 1 is perfect), then
averaged. Data, prompts and grading follow nanochat's `scripts/chat_eval.py` and `tasks/`:

- ARC-Easy, ARC-Challenge and MMLU are multiple choice, written with nanochat's
  `render_mc`. The answer is whichever choice letter the model rates most likely next,
  among the letters offered (baseline 25%).
- GSM8K is math word problems. The model writes a solution greedily, up to 512 tokens,
  and is right if the number after its "####" matches the reference (baseline 0%).
- HumanEval is Python functions. The model writes code greedily; it is right if the code
  passes the task's tests (baseline 0%).
- ChatCORE averages all five; ChatCORE_cat averages the three multiple-choice ones.

Prompts take one of two forms. A fine-tuned run (`sft.py`) has nanochat's chat tokens, so
each problem is written exactly as nanochat writes it (`chat_format.render_for_completion`),
letters are scored without a space, and generation stops at <|assistant_end|>, with the
calculator's results written in as nanochat's Engine does. A base model has no chat
tokens, so each problem is written as the chat page's plain transcript
(`chat_web.transcript`): a preamble, "User: <question>", then "Assistant:". Its answer is
what follows, up to the point where it starts the user's next turn. That score is a
baseline from before fine-tuning, which nanochat never reports.

Multiple-choice prompts are scored in batches packed to a token budget, like CORE.
Generations run in batches through `generate.greedy_batch`. HumanEval runs the model's
code in a subprocess with a timeout and memory limit, which isn't a sandbox, so run
this on a throwaway machine such as a Colab VM.

    python -m toylib_projects.p02_gpt_web.scripts.chat_eval --run d20_nanochat \\
        --checkpoint-dir gs://tinystories-checkpoints/toylib-p02-climbmix/d20_nanochat \\
        --output chatcore_d20.json --completions chatcore_d20.jsonl

Datasets come from the Hugging Face Hub's parquet exports, downloaded once to
data/chat_eval, and are shuffled with nanochat's seed so `--max-problems` picks the
same problems as nanochat's.
"""

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import typing
import urllib.request

import numpy as np

from toylib.data import download
from toylib_projects.p02_gpt_web.scripts import core_eval
from toylib_projects.p02_gpt_web.tokenizer import bpe

DATA_DIR = pathlib.Path(__file__).parents[1] / "data" / "chat_eval"
HUB_PARQUET = "https://huggingface.co/api/datasets/{repo}/parquet/{subset}/{split}"
TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")
CATEGORICAL = ("ARC-Easy", "ARC-Challenge", "MMLU")
BASELINES = {"ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25, "GSM8K": 0.0, "HumanEval": 0.0}
BOS = bpe.EOT_ID  # Pads batches too: padded positions are never read.
STOP = "\nUser:"  # Where the model starts the user's next turn: the end of its answer.

# last_logits(tokens [batch, time], positions [batch]) -> logits [batch, vocab] at those positions.
LastLogits = typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
# complete(prompts) -> each prompt's greedy continuation, as text cut before STOP.
Complete = typing.Callable[[list[list[int]]], list[str]]


@dataclasses.dataclass(frozen=True)
class Problem:
    question: str  # The user's message.
    answer: str  # The right letter, GSM8K's number, or HumanEval's canonical solution.
    letters: tuple[str, ...] = ()  # Multiple choice only.
    entry_point: str = ""  # HumanEval only.
    test: str = ""  # HumanEval only.


# --- Data ------------------------------------------------------------------------------


def load_hub(repo: str, subset: str, split: str, data_dir: pathlib.Path = DATA_DIR) -> list[dict]:
    """A Hub dataset's rows from its parquet export, in nanochat's shuffled order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = data_dir / repo.replace("/", "--") / subset / split
    manifest = folder / "manifest.json"
    if not manifest.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HUB_PARQUET.format(repo=repo, subset=subset, split=split)) as r:
            urls = json.loads(r.read())
        names = [f"{i:05d}.parquet" for i in range(len(urls))]
        for url, name in zip(urls, names):
            download.fetch(url, folder / name)
        manifest.write_text(json.dumps(names))  # Written last: its presence means complete.
    names = json.loads(manifest.read_text())
    rows = pa.concat_tables([pq.read_table(folder / n) for n in names]).to_pylist()
    # nanochat's HubDataset.shuffle(seed=42), which matches datasets.Dataset.shuffle.
    return [rows[i] for i in np.random.default_rng(42).permutation(len(rows))]


def render_mc(question: str, letters: typing.Sequence[str], choices: typing.Sequence[str]) -> str:
    """nanochat's multiple-choice format, with each letter after its choice."""
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    return query + "\nRespond only with the letter of the correct answer."


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def gsm8k_number(text: str) -> str | None:
    """The number after "####", without commas, as nanochat and OpenAI's grader read it."""
    match = GSM_RE.search(text)
    return match.group(1).strip().replace(",", "") if match else None


def load_task(name: str, data_dir: pathlib.Path = DATA_DIR) -> list[Problem]:
    if name in ("ARC-Easy", "ARC-Challenge"):
        return [
            Problem(
                render_mc(r["question"], r["choices"]["label"], r["choices"]["text"]),
                r["answerKey"],
                letters=tuple(r["choices"]["label"]),
            )
            for r in load_hub("allenai/ai2_arc", name, "test", data_dir)
        ]
    if name == "MMLU":
        letters = ("A", "B", "C", "D")
        return [
            Problem(render_mc(r["question"], letters, r["choices"]), letters[r["answer"]], letters=letters)
            for r in load_hub("cais/mmlu", "all", "test", data_dir)
        ]
    if name == "GSM8K":
        return [
            Problem(r["question"], gsm8k_number(r["answer"]))
            for r in load_hub("openai/gsm8k", "main", "test", data_dir)
        ]
    if name == "HumanEval":
        return [
            Problem(r["prompt"], f"{r['prompt']}\n{r['canonical_solution']}", entry_point=r["entry_point"], test=r["test"])
            for r in load_hub("openai/openai_humaneval", "openai_humaneval", "test", data_dir)
        ]
    raise ValueError(f"Unknown task {name!r}; expected one of {TASKS}")


# --- Prompts ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Transcript:
    """A base model's prompts: the problem as the chat page's plain transcript."""

    encode: typing.Callable[[str], list[int]]
    bos: int
    choice_prefix: str = " "  # After "Assistant:" an answer is " A".

    def prompt(self, question: str, max_len: int) -> list[int]:
        """The problem as a one-turn transcript, ending where the assistant's reply begins."""
        from toylib_projects.p02_gpt_web.scripts import chat_web

        text = chat_web.transcript([{"role": "user", "content": question}])
        return ([self.bos] + self.encode(text))[-max_len:]


@dataclasses.dataclass(frozen=True)
class ChatTokens:
    """A fine-tuned model's prompts, in nanochat's chat format."""

    tokenizer: typing.Any  # chat_format.ChatTokenizer
    choice_prefix: str = ""  # After <|assistant_start|> an answer is "A".

    @property
    def encode(self) -> typing.Callable[[str], list[int]]:
        return self.tokenizer.encode

    def prompt(self, question: str, max_len: int) -> list[int]:
        from toylib_projects.p02_gpt_web import chat_format

        ids = chat_format.render_for_completion(self.tokenizer, [{"role": "user", "content": question}])
        return ids[-max_len:]


Format = Transcript | ChatTokens


def choice_tokens(fmt: Format, letters: typing.Sequence[str]) -> tuple[list[int], list[int]]:
    """The tokens all answers share, then the one token that tells each letter apart.

    ARC sometimes labels choices "1".."4", and our tokenizer writes " 1" as a space then a
    digit, so in a transcript the shared space joins the prompt.
    """
    ids = [fmt.encode(fmt.choice_prefix + letter) for letter in letters]
    shared = core_eval.common_length(ids)
    if any(len(x) != shared + 1 for x in ids):
        raise ValueError(f"Choices {letters} differ in more than one token: {ids}")
    return ids[0][:shared], [x[shared] for x in ids]


# --- Scoring ---------------------------------------------------------------------------


def categorical_correct(
    last_logits: LastLogits, problems: list[Problem], fmt: Format, max_len: int, token_budget: int
) -> list[bool]:
    """Whether the model rates the right letter highest among each problem's letters."""
    prompts, options = [], []
    for p in problems:
        shared, letter_ids = choice_tokens(fmt, p.letters)
        prompts.append(fmt.prompt(p.question, max_len - len(shared)) + shared)
        options.append(letter_ids)
    correct = [False] * len(problems)
    order = sorted(range(len(problems)), key=lambda i: len(prompts[i]))
    start = 0
    while start < len(order):
        width = min(core_eval.bucket(len(prompts[order[start]])), max_len)
        rows = max(1, token_budget // width)
        batch = [i for i in order[start : start + rows] if len(prompts[i]) <= width]
        start += len(batch)
        tokens = np.full((len(batch), width), BOS, dtype=np.int32)
        for row, i in enumerate(batch):
            tokens[row, : len(prompts[i])] = prompts[i]
        logits = last_logits(tokens, np.array([len(prompts[i]) - 1 for i in batch]))
        for row, i in enumerate(batch):
            pick = int(np.argmax(logits[row, options[i]]))
            correct[i] = problems[i].letters[pick] == problems[i].answer
    return correct


def extract_program(completion: str) -> str:
    """The first ``` code block, or the whole completion: nanochat's HumanEval rule."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
    return blocks[0].strip() if blocks else completion.strip()


def extract_imports(prompt: str) -> str:
    """The import lines at the top of a HumanEval prompt."""
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def _limit_memory():  # Runs in the child before exec. Not available on every OS.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    except (ImportError, ValueError, OSError):
        pass


def passes_tests(problem: Problem, completion: str, timeout: float = 10.0) -> bool:
    """Whether the completion's code passes the HumanEval problem's tests."""
    program = "\n\n".join([
        extract_imports(problem.question),
        extract_program(completion),
        problem.test,
        f"check({problem.entry_point})\n",
    ])
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program], cwd=folder, timeout=timeout,
                stdin=subprocess.DEVNULL, capture_output=True, preexec_fn=_limit_memory,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def grade(task: str, problem: Problem, completion: str) -> bool:
    if task == "GSM8K":
        return gsm8k_number(completion) == problem.answer
    if task == "HumanEval":
        return passes_tests(problem, completion)
    raise ValueError(f"{task} is not generative")


def generative_results(
    complete: Complete, task: str, problems: list[Problem], fmt: Format, max_len: int, batch_size: int,
    verbose: bool = True,
) -> list[tuple[str, bool]]:
    """Each problem's completion and whether it is right, generated in length-sorted batches."""
    prompts = [fmt.prompt(p.question, max_len) for p in problems]
    order = sorted(range(len(problems)), key=lambda i: len(prompts[i]))
    out: list[tuple[str, bool]] = [("", False)] * len(problems)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        for i, text in zip(batch, complete([prompts[i] for i in batch])):
            out[i] = (text, grade(task, problems[i], text))
        if verbose:
            print(f"  {task}: {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def centered(accuracy: float, baseline: float) -> float:
    return (accuracy - baseline) / (1.0 - baseline)


def summarize(accuracy: dict[str, float]) -> dict:
    """Accuracies, centered scores, and ChatCORE where every task needed was run."""
    out = {"accuracy": accuracy, "centered": {t: centered(a, BASELINES[t]) for t, a in accuracy.items()}}
    if all(t in accuracy for t in CATEGORICAL):
        out["ChatCORE_cat"] = float(np.mean([out["centered"][t] for t in CATEGORICAL]))
    if all(t in accuracy for t in TASKS):
        out["ChatCORE"] = float(np.mean([out["centered"][t] for t in TASKS]))
    return out


def evaluate(
    last_logits: LastLogits,
    complete: Complete,
    fmt: Format,
    max_len: int,
    tasks: typing.Sequence[str] = TASKS,
    max_problems: dict[str, int | None] | int | None = None,
    token_budget: int = 16_384,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Run the tasks; returns `summarize`'s scores and every generated answer.

    `max_problems` caps each task, or each task named in a dict: nanochat checks progress
    during fine-tuning on every multiple-choice problem but 24 of each generative task.
    """
    accuracy, completions = {}, []
    for task in tasks:
        limit = max_problems.get(task) if isinstance(max_problems, dict) else max_problems
        problems = load_task(task)[:limit]
        start = time.perf_counter()
        if task in CATEGORICAL:
            correct = categorical_correct(last_logits, problems, fmt, max_len, token_budget)
        else:
            results = generative_results(
                complete, task, problems, fmt, max_len - max_new_tokens, batch_size, verbose
            )
            correct = [ok for _, ok in results]
            completions += [
                dict(task=task, index=i, question=p.question, completion=text, correct=ok)
                for i, (p, (text, ok)) in enumerate(zip(problems, results))
            ]
        accuracy[task] = float(np.mean(correct))
        if verbose:
            print(f"{task} accuracy: {accuracy[task]:.4f} ({sum(correct)}/{len(correct)}, "
                  f"{time.perf_counter() - start:.0f} s)", flush=True)
    return summarize(accuracy), completions


# --- Model -----------------------------------------------------------------------------


def model_functions(model, tokenizer, max_new_tokens: int):
    """(last_logits, complete, fmt) for a model, in the format its tokenizer implies."""
    import jax
    import jax.numpy as jnp

    from toylib_projects.p02_gpt_web import chat_format, generate

    @jax.jit
    def run_last_logits(model, tokens, positions):
        hidden = model.hidden(tokens)
        last = jnp.take_along_axis(hidden, positions[:, None, None], axis=1)[:, 0]
        return model.logits(last).astype(jnp.float32)

    def last_logits(tokens, positions):
        return np.asarray(run_last_logits(model, jnp.asarray(tokens), jnp.asarray(positions)))

    bos = BOS
    if isinstance(tokenizer, chat_format.ChatTokenizer):
        fmt = ChatTokens(tokenizer)

        def complete(prompts):
            outs = generate.greedy_batch(
                model, prompts, max_new_tokens, stop_tokens={tokenizer.assistant_end, bos},
                tool=lambda ids: chat_format.tool_tokens(tokenizer, ids),
            )
            return [tokenizer.decode(ids) for ids in outs]
    else:
        fmt = Transcript(tokenizer.encode, bos)

        def complete(prompts):
            outs = generate.greedy_batch(
                model, prompts, max_new_tokens, stop_tokens={bos},
                is_done=lambda ids: STOP in tokenizer.decode(ids),
            )
            return [tokenizer.decode(ids).split(STOP)[0] for ids in outs]

    return last_logits, complete, fmt


def load_run(run_name: str, checkpoint_dir: str, step: int | None, max_new_tokens: int):
    """A run's model as (last_logits, complete, fmt, max_len). Fine-tuned runs are in sft.RUNS."""
    from toylib_projects.p02_gpt_web import generate

    model, tokenizer = generate.load(run_name, checkpoint_dir, step)
    return (*model_functions(model, tokenizer, max_new_tokens), model.config.seq_len)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="d20_nanochat", help="A key of train.RUNS or sft.RUNS")
    parser.add_argument("--checkpoint-dir", required=True, help="Local or gs:// directory of checkpoints")
    parser.add_argument("--step", type=int, default=None, help="Default: the latest")
    parser.add_argument("--tasks", default="|".join(TASKS), help="Tasks to run, joined by |")
    parser.add_argument("--max-problems", type=int, default=None, help="Per task. Default: all")
    parser.add_argument("--token-budget", type=int, default=16_384, help="Tokens per multiple-choice batch")
    parser.add_argument("--batch-size", type=int, default=32, help="Rows per generation batch")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=pathlib.Path, help="JSON of the scores")
    parser.add_argument("--completions", type=pathlib.Path, help="JSONL of every generated answer")
    args = parser.parse_args()

    last_logits, complete, fmt, max_len = load_run(
        args.run, args.checkpoint_dir, args.step, args.max_new_tokens
    )
    out, completions = evaluate(
        last_logits, complete, fmt, max_len, args.tasks.split("|"), args.max_problems,
        args.token_budget, args.batch_size, args.max_new_tokens,
    )
    for key in ("ChatCORE_cat", "ChatCORE"):
        if key in out:
            print(f"{key}: {out[key]:.4f}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
    if args.completions:
        args.completions.write_text("".join(json.dumps(c) + "\n" for c in completions))


if __name__ == "__main__":
    main()
