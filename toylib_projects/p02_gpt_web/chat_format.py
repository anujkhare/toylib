"""nanochat's chat format: conversations written between special tokens.

nanochat reserves nine special tokens in its tokenizer from the start. Ours was trained
with only <|endoftext|>, which starts every document and so plays nanochat's <|bos|>, and
its 32,768 ids are all taken. Fine-tuning therefore appends nanochat's other eight after
them, and the model grows a row for each in its embeddings and output layer (see
`sft.py`).

`render_conversation` and `render_for_completion` follow nanochat's `tokenizer.py`. A
conversation is a dict with "messages": a list of {"role", "content"}, alternating user
and assistant, optionally after a system message. An assistant's content is a string, or
a list of parts: {"type": "text" | "python" | "python_output", "text": ...}.
"""

import copy
import typing

import numpy as np

from toylib_projects.p02_gpt_web.tokenizer import bpe

BOS = bpe.EOT_ID
SPECIAL_TOKENS = (
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",  # The assistant calls the calculator...
    "<|python_end|>",
    "<|output_start|>",  # ...and its result is written in, not predicted.
    "<|output_end|>",
)
MAX_TOKENS = 2048  # nanochat's cap on a rendered conversation.


class ChatTokenizer:
    """A base tokenizer with nanochat's chat tokens after its last id."""

    bos_token_id = BOS
    eos_token_id = BOS

    def __init__(self, base: bpe.Tokenizer):
        self.base = base
        self.base_size = len(base)
        self.special = {name: self.base_size + i for i, name in enumerate(SPECIAL_TOKENS)}
        self.names = {i: name for name, i in self.special.items()}

    def __len__(self) -> int:
        return self.base_size + len(SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def __getattr__(self, name: str) -> int:
        """`tokenizer.user_start` and so on: the id of <|user_start|>."""
        special = self.__dict__.get("special", {})
        if f"<|{name}|>" in special:
            return special[f"<|{name}|>"]
        raise AttributeError(name)

    def __call__(self, texts: list[str], **_) -> dict[str, list[list[int]]]:
        return self.base(texts)

    def encode(self, text: str) -> list[int]:
        """Text as base tokens. Special tokens written as text stay text, as in nanochat."""
        return self.base.encode(text)

    def decode(self, ids: typing.Iterable[int]) -> str:
        """Text, with each special token written as its name, as tiktoken decodes them."""
        out, run = [], []
        for i in ids:
            i = int(i)
            if i in self.names:
                out += [self.base.decode(run), self.names[i]]
                run = []
            else:
                run.append(i)
        return "".join(out) + self.base.decode(run)

    def bytes_per_token(self) -> np.ndarray:
        """-1 for special tokens, so bits per byte leaves them out."""
        return np.concatenate([self.base.bytes_per_token(), -np.ones(len(SPECIAL_TOKENS), np.int32)])


def render_conversation(
    tokenizer: ChatTokenizer, conversation: dict, max_tokens: int = MAX_TOKENS
) -> tuple[list[int], list[int]]:
    """A conversation's tokens, and a mask of 1 where the assistant's tokens are trained.

    The assistant learns its text, its calculator calls, and <|assistant_end|>; never the
    user's turns, the calculator's results, or the tokens that open a turn.
    """
    ids, mask = [], []

    def add(tokens, value):
        tokens = [tokens] if isinstance(tokens, int) else tokens
        ids.extend(tokens)
        mask.extend([value] * len(tokens))

    messages = conversation["messages"]
    if messages[0]["role"] == "system":  # Folded into the first user message.
        messages = copy.deepcopy(messages)
        assert messages[1]["role"] == "user", "A system message must come before a user's"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    assert messages, "A conversation needs a message"

    t = tokenizer
    add(BOS, 0)
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected, f"Message {i} is from {message['role']}, not {expected}"
        content = message["content"]
        if message["role"] == "user":
            assert isinstance(content, str), "User messages are strings"
            add(t.user_start, 0)
            add(t.encode(content), 0)
            add(t.user_end, 0)
            continue
        add(t.assistant_start, 0)
        if isinstance(content, str):
            add(t.encode(content), 1)
        else:
            for part in content:
                text = t.encode(part["text"])
                if part["type"] == "text":
                    add(text, 1)
                elif part["type"] == "python":
                    add(t.python_start, 1)
                    add(text, 1)
                    add(t.python_end, 1)
                elif part["type"] == "python_output":
                    add(t.output_start, 0)
                    add(text, 0)
                    add(t.output_end, 0)
                else:
                    raise ValueError(f"Unknown part type {part['type']!r}")
        add(t.assistant_end, 1)
    return ids[:max_tokens], mask[:max_tokens]


def render_for_completion(tokenizer: ChatTokenizer, messages: list[dict]) -> list[int]:
    """The conversation so far, ending where the assistant's next reply begins.

    `messages` ends with a user's turn (nanochat's version drops a final assistant turn
    first; callers here never pass one).
    """
    ids, _ = render_conversation(tokenizer, {"messages": messages}, max_tokens=10**9)
    return ids + [tokenizer.assistant_start]


def calculator(expr: str) -> typing.Any:
    """nanochat's `use_calculator`: arithmetic, or a string's `.count()`, else None.

    nanochat also stops an evaluation after 3 seconds. Without `**` neither form can run
    long, and a signal-based timer only works on the main thread, so there's none here.
    """
    expr = expr.replace(",", "")
    if all(c in "0123456789*+-/.() " for c in expr):
        if "**" in expr:
            return None
    else:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
        banned = ("__", "import", "exec", "eval", "compile", "open", "file", "input",
                  "raw_input", "globals", "locals", "vars", "dir", "getattr", "setattr",
                  "delattr", "hasattr")
        if not all(c in allowed for c in expr) or any(b in expr.lower() for b in banned):
            return None
        if ".count(" not in expr:
            return None
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def tool_tokens(tokenizer: ChatTokenizer, generated: typing.Sequence[int]) -> list[int]:
    """What the calculator writes after the token just generated, as nanochat's Engine does.

    When the assistant closes a calculator call, the expression since <|python_start|> is
    evaluated and its result is forced in between <|output_start|> and <|output_end|>.
    """
    t = tokenizer
    if not generated or generated[-1] != t.python_end:
        return []
    starts = [i for i, x in enumerate(generated[:-1]) if x == t.python_start]
    if not starts or t.python_end in generated[starts[-1] + 1 : -1]:
        return []
    result = calculator(t.decode(generated[starts[-1] + 1 : -1]))
    if result is None:
        return []
    return [t.output_start, *t.encode(str(result)), t.output_end]
