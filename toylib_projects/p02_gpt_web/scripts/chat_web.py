"""A local web page for talking to a trained run, like nanochat's chat_web.

    python -m toylib_projects.p02_gpt_web.scripts.chat_web \\
        --checkpoint-dir toylib_projects/p02_gpt_web/outputs/d20_nanochat/checkpoints

then open http://localhost:8000. The server streams tokens as server-sent events from
POST /generate, so the same endpoint can later sit behind a hosted API.

A fine-tuned run (`--run d20_sft`) chats in nanochat's format, between chat tokens, and
replies until <|assistant_end|>. A base model was never fine-tuned on conversations and has
no chat tokens, so chat mode writes the conversation as a plain transcript,
"User: ... / Assistant: ...", and stops when the model starts the user's next turn.
Complete mode continues raw text either way. The page doesn't run the calculator, so a
fine-tuned model's calculator calls show as written and it guesses their results.
"""

import argparse
import json
import pathlib
import threading
import time
import typing
from http import server

import jax.numpy as jnp

from toylib_projects.p02_gpt_web import chat_format, generate, tokenize_webtext

PAGE = pathlib.Path(__file__).with_name("chat.html")
EOT = tokenize_webtext.EOT_ID
PREAMBLE = (
    "The following is a conversation between a curious user and a friendly, knowledgeable "
    "assistant. The assistant answers clearly and accurately.\n\n"
)
STOP = "\nUser:"


def transcript(messages: list[dict]) -> str:
    """The conversation as text, ending where the assistant's next reply begins."""
    turns = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}" for m in messages]
    return PREAMBLE + "\n".join(turns) + "\nAssistant:"


def is_chat(tokenizer) -> bool:
    """Whether the run was fine-tuned with chat tokens."""
    return isinstance(tokenizer, chat_format.ChatTokenizer)


def fit(tokenizer, messages: list[dict], budget: int) -> list[int]:
    """The conversation's tokens, dropping the oldest turns until it fits `budget`."""
    while True:
        if is_chat(tokenizer):
            ids = chat_format.render_for_completion(tokenizer, messages)
        else:
            ids = [EOT] + tokenizer.encode(transcript(messages))
        if len(ids) <= budget or len(messages) <= 1:
            return ids[-budget:]
        messages = messages[2:] if len(messages) > 2 else messages[1:]


def held_back(text: str, stop: str) -> int:
    """How many trailing characters might begin `stop`, so they wait for the next token."""
    for n in range(min(len(stop), len(text)), 0, -1):
        if stop.startswith(text[-n:]):
            return n
    return 0


def stream_text(
    gen: generate.Generator,
    tokenizer,
    prompt: list[int],
    stop: str | None,
    stop_tokens: typing.Container[int] = (EOT,),
    **kwargs,
) -> typing.Iterator[str]:
    """Decoded text as it grows, never splitting a UTF-8 character or emitting `stop`."""
    ids, sent = [], 0
    for token in gen.stream(prompt, stop_tokens=stop_tokens, **kwargs):
        ids.append(token)
        text = tokenizer.decode(ids)
        if stop and stop in text:
            text = text[: text.index(stop)]
            if len(text) > sent:
                yield text[sent:]
            return
        ready = len(text) - (held_back(text, stop) if stop else 0)
        # A token may end inside a multi-byte character, which decodes as U+FFFD.
        while ready > sent and text[ready - 1] == "�":
            ready -= 1
        if ready > sent:
            yield text[sent:ready]
            sent = ready
    text = tokenizer.decode(ids)
    if stop and stop in text:
        text = text[: text.index(stop)]
    if len(text) > sent:
        yield text[sent:]


class Handler(server.BaseHTTPRequestHandler):
    gen: generate.Generator
    tokenizer: typing.Any
    run_name: str
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # Quieter than the default.
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            info = dict(run=self.run_name, context=self.gen.seq_len, chat_tokens=is_chat(self.tokenizer))
            self._send(200, json.dumps(info).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, b"Not found", "text/plain")
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            max_tokens = max(1, min(int(request.get("max_tokens", 256)), 1024))
            options = dict(
                max_new_tokens=max_tokens,
                temperature=max(0.0, float(request.get("temperature", 0.8))),
                top_k=int(request["top_k"]) if request.get("top_k") else None,
                seed=int(request.get("seed", time.time_ns() % 2**31)),
            )
            budget = self.gen.seq_len - max_tokens
            stop_tokens = {EOT}
            if request.get("mode") == "complete":
                prompt = ([EOT] + self.tokenizer.encode(request["prompt"]))[-budget:]
                stop = None
            elif is_chat(self.tokenizer):
                prompt, stop = fit(self.tokenizer, request["messages"], budget), None
                stop_tokens = {EOT, self.tokenizer.assistant_end}
            else:
                prompt, stop = fit(self.tokenizer, request["messages"], budget), STOP
        except (KeyError, TypeError, ValueError) as e:
            return self._send(400, f"Bad request: {e}".encode(), "text/plain")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with self.lock:  # One generation at a time: the model is the bottleneck.
            start = time.perf_counter()
            try:
                for piece in stream_text(self.gen, self.tokenizer, prompt, stop, stop_tokens, **options):
                    self._event({"text": piece})
                elapsed = time.perf_counter() - start
                self._event({"done": True, "prompt_tokens": len(prompt), "seconds": round(elapsed, 2)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # The page stopped reading: stop generating.

    def _event(self, data: dict) -> None:
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="d20_nanochat")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None,
                        help="Compute dtype. Default: float32 on a CPU, else the run's")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    dtype = getattr(jnp, args.dtype) if args.dtype else None
    model, tokenizer = generate.load(args.run, str(args.checkpoint_dir.resolve()), args.step, dtype)
    gen = generate.Generator(model)
    print("Compiling...", flush=True)
    list(gen.stream([EOT], max_new_tokens=2))
    Handler.gen, Handler.tokenizer, Handler.run_name = gen, tokenizer, args.run
    httpd = server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.run} on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
