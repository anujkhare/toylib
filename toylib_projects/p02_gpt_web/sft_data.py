"""nanochat's fine-tuning data, rendered in its chat format and packed into rows.

The mixture follows nanochat's `scripts/chat_sft.py`:

    train: SmolTalk (460K general conversations), MMLU's auxiliary_train three times
           (100K multiple-choice questions, to teach the answer format) and GSM8K's train
           split four times (7.5K math problems that use the calculator).
    val:   SmolTalk's test split, the first 5,200 MMLU test questions and the first 420
           GSM8K test problems, in about the train mixture's proportions.

Rows are packed as nanochat's `sft_data_generator_bos_bestfit` packs them: each row
starts a conversation, then takes the longest of the next 100 conversations that still
fits, and is padded once none does. No conversation is cut. Padding is masked out of the
loss, as are all but the assistant's tokens (`chat_format.render_conversation`).

Rendering is slow in Python, so `prepare` does it once and writes:

    <data_dir>/sft_<split>_<vocab>.npy       uint16 tokens, [rows, seq_len + 1]
    <data_dir>/sft_<split>_<vocab>_mask.npy  uint8 loss mask of the same shape

Run once with:
    uv run python -m toylib_projects.p02_gpt_web.sft_data
"""

import argparse
import dataclasses
import json
import multiprocessing
import pathlib
import random
import re
import time
import typing

import numpy as np

from toylib_projects.p02_gpt_web import chat_format
from toylib_projects.p02_gpt_web import tokenize_webtext
from toylib_projects.p02_gpt_web.scripts import chat_eval

DATA_DIR = pathlib.Path(__file__).parent / "data" / "sft"
SEQ_LEN = 2048
BUFFER_SIZE = 100  # Conversations the packer chooses among.
MMLU_EPOCHS = 3
GSM8K_EPOCHS = 4

Conversation = dict[str, typing.Any]


# --- Tasks, as nanochat's tasks/ write them ---------------------------------------------


def smoltalk(split: str) -> list[Conversation]:
    rows = chat_eval.load_hub("HuggingFaceTB/smol-smoltalk", "default", split)
    return [{"messages": r["messages"]} for r in rows]


def mmlu(split: str, stop: int | None = None) -> list[Conversation]:
    letters = ("A", "B", "C", "D")
    rows = chat_eval.load_hub("cais/mmlu", "all", split)[:stop]
    return [
        {"messages": [
            {"role": "user", "content": chat_eval.render_mc(r["question"], letters, r["choices"])},
            {"role": "assistant", "content": letters[r["answer"]]},
        ]}
        for r in rows
    ]


def gsm8k_parts(answer: str) -> list[dict]:
    """A GSM8K solution as text and calculator parts: "<<12/60=0.2>>" is a call and its result."""
    parts = []
    for piece in re.split(r"(<<[^>]+>>)", answer):
        if piece.startswith("<<") and piece.endswith(">>"):
            expr, result = piece[2:-2].rsplit("=", 1) if "=" in piece else (piece[2:-2], "")
            parts += [{"type": "python", "text": expr}, {"type": "python_output", "text": result}]
        else:
            parts.append({"type": "text", "text": piece})
    return parts


def gsm8k(split: str, stop: int | None = None) -> list[Conversation]:
    rows = chat_eval.load_hub("openai/gsm8k", "main", split)[:stop]
    return [
        {"messages": [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": gsm8k_parts(r["answer"])},
        ]}
        for r in rows
    ]


def mixture_order(lengths: list[int], seed: int = 42) -> list[tuple[int, int]]:
    """nanochat's TaskMixture: every (task, index) pair, shuffled together."""
    order = [(task, i) for task, n in enumerate(lengths) for i in range(n)]
    random.Random(seed).shuffle(order)
    return order


def splits() -> dict[str, list[list[Conversation]]]:
    """Each split's tasks. A task listed twice is an extra epoch of it."""
    mmlu_train, gsm8k_train = mmlu("auxiliary_train"), gsm8k("train")
    return {
        "train": [smoltalk("train"), *[mmlu_train] * MMLU_EPOCHS, *[gsm8k_train] * GSM8K_EPOCHS],
        "val": [smoltalk("test"), mmlu("test", stop=5200), gsm8k("test", stop=420)],
    }


# --- Rendering ------------------------------------------------------------------------

_worker_tokenizer: chat_format.ChatTokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    from toylib_projects.p02_gpt_web.tokenizer import bpe

    _worker_tokenizer = chat_format.ChatTokenizer(bpe.Tokenizer.load(pathlib.Path(tokenizer_path)))


def _render(conversation: Conversation) -> tuple[np.ndarray, np.ndarray]:
    ids, mask = chat_format.render_conversation(_worker_tokenizer, conversation, SEQ_LEN)
    return np.array(ids, np.uint16), np.array(mask, np.uint8)


def render_tasks(
    tasks: list[list[Conversation]], tokenizer_path: pathlib.Path, workers: int
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Each task's conversations rendered, once per distinct task."""
    rendered: dict[int, list] = {}
    with multiprocessing.Pool(workers, _init_worker, (str(tokenizer_path),)) as pool:
        for task in tasks:
            if id(task) not in rendered:
                start = time.perf_counter()
                rendered[id(task)] = pool.map(_render, task, chunksize=256)
                print(f"  rendered {len(task):,} conversations in {time.perf_counter() - start:.0f} s", flush=True)
    return [rendered[id(task)] for task in tasks]


# --- Packing --------------------------------------------------------------------------


def pack(
    conversations: typing.Iterable[tuple[np.ndarray, np.ndarray]],
    row_len: int,
    bos: int = chat_format.BOS,
    buffer_size: int = BUFFER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit packing into rows of `row_len` tokens, until the conversations run out.

    The last rows, once fewer than `buffer_size` conversations are left, are packed from
    what remains. Padding repeats `bos` and is masked.
    """
    source = iter(conversations)
    buffer: list[tuple[np.ndarray, np.ndarray]] = []
    exhausted = False
    rows, masks = [], []
    while True:
        while not exhausted and len(buffer) < buffer_size:
            try:
                buffer.append(next(source))
            except StopIteration:
                exhausted = True
        if not buffer:
            break
        row = np.full(row_len, bos, np.uint16)
        mask = np.zeros(row_len, np.uint8)
        used = 0
        while True:
            remaining = row_len - used
            best, best_len = -1, 0
            for i, (ids, _) in enumerate(buffer):
                if best_len < len(ids) <= remaining:
                    best, best_len = i, len(ids)
            if best < 0:
                break
            ids, m = buffer.pop(best)
            row[used : used + len(ids)], mask[used : used + len(ids)] = ids, m
            used += len(ids)
            while not exhausted and len(buffer) < buffer_size:
                try:
                    buffer.append(next(source))
                except StopIteration:
                    exhausted = True
        rows.append(row)
        masks.append(mask)
    return np.stack(rows), np.stack(masks)


def paths(split: str, vocab_size: int, data_dir: pathlib.Path = DATA_DIR) -> tuple[pathlib.Path, pathlib.Path]:
    return data_dir / f"sft_{split}_{vocab_size}.npy", data_dir / f"sft_{split}_{vocab_size}_mask.npy"


def prepare(
    tokenizer_path: pathlib.Path, data_dir: pathlib.Path = DATA_DIR, workers: int = 8, seq_len: int = SEQ_LEN
) -> dict:
    """Render and pack both splits, and write them with a summary of what they hold."""
    from toylib_projects.p02_gpt_web.tokenizer import bpe

    vocab_size = len(chat_format.ChatTokenizer(bpe.Tokenizer.load(tokenizer_path)))
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, tasks in splits().items():
        print(f"{split}: {len(tasks)} tasks, {sum(map(len, tasks)):,} conversations", flush=True)
        rendered = render_tasks(tasks, tokenizer_path, workers)
        order = mixture_order([len(t) for t in tasks])
        rows, masks = pack((rendered[t][i] for t, i in order), seq_len + 1)
        tokens_path, mask_path = paths(split, vocab_size, data_dir)
        np.save(tokens_path, rows)
        np.save(mask_path, masks)
        content = sum(len(ids) for task in rendered for ids, _ in task)
        summary[split] = dict(
            conversations=len(order), rows=len(rows), tokens=int(content),
            trained_tokens=int(masks[:, 1:].sum()), fill=content / rows.size,
        )
        print(f"  {json.dumps(summary[split])}", flush=True)
    (data_dir / f"sft_{vocab_size}.json").write_text(json.dumps(summary, indent=2))
    return summary


# --- Reading --------------------------------------------------------------------------


@dataclasses.dataclass
class PackedRowsDataset:
    """Batches of packed rows, in order; a pass over them is an epoch, then it repeats.

    Like `data.TokenBinDataset`, a batch is a function of its index alone, so resuming
    needs only the index, and `repeatable=True` restarts every `iter()` for validation.
    """

    path: str
    mask_path: str
    batch_size: int
    repeatable: bool = False
    tokenizer: typing.Any = None  # For the experiment's sampling.

    def __post_init__(self):
        self.tokens = np.load(self.path, mmap_mode="r")
        self.mask = np.load(self.mask_path, mmap_mode="r")
        if self.tokens.shape != self.mask.shape:
            raise ValueError(f"Tokens {self.tokens.shape} and mask {self.mask.shape} differ")
        if len(self.tokens) < self.batch_size:
            raise ValueError(f"{self.path} has {len(self.tokens)} rows, fewer than a batch")
        self._index = 0

    @property
    def steps_per_epoch(self) -> int:
        return len(self.tokens) // self.batch_size  # The last partial batch is dropped.

    def _batch(self, index: int) -> dict[str, np.ndarray]:
        start = (index % self.steps_per_epoch) * self.batch_size
        rows = np.asarray(self.tokens[start : start + self.batch_size], np.int32)
        mask = np.asarray(self.mask[start : start + self.batch_size, 1:], np.int32)
        return {"inputs": rows[:, :-1], "targets": rows[:, 1:], "mask": mask}

    def __iter__(self) -> typing.Iterator[dict[str, np.ndarray]]:
        if self.repeatable:
            self._index = 0
        while True:
            self._index += 1
            yield self._batch(self._index - 1)

    def get_state(self) -> dict[str, int]:
        return {"index": self._index}

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        self._index = int(state["index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", type=pathlib.Path,
                        default=tokenize_webtext.tokenizer_path(32_768, tokenize_webtext.data_dir_for(120)))
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    prepare(args.tokenizer, args.data_dir, args.workers)


if __name__ == "__main__":
    main()
