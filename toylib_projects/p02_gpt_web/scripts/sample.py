"""Sample from a trained run, as nanochat's base_eval does, and time generation.

    python -m toylib_projects.p02_gpt_web.scripts.sample --run d20_nanochat \\
        --checkpoint-dir toylib_projects/p02_gpt_web/outputs/d20_nanochat/checkpoints

Prints greedy completions of nanochat's prompts, a few samples at temperature 1, and with
--benchmark, prefill and decoding speed.
"""

import argparse
import pathlib
import time

import jax.numpy as jnp
import numpy as np

from toylib_projects.p02_gpt_web import generate, tokenize_webtext

# nanochat's base_eval prompts.
PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
EOT = tokenize_webtext.EOT_ID


def complete(gen, tokenizer, text, **kwargs) -> str:
    ids = [EOT] + tokenizer.encode(text)
    return tokenizer.decode(list(gen.stream(ids, stop_tokens={EOT}, **kwargs)))


def benchmark(gen, tokenizer, contexts=(16, 512, 1536), new_tokens=64) -> list[dict]:
    """Decoding tokens per second after prompts of a few lengths, and prefill speed."""
    rows = []
    text = "The history of the city goes back many centuries. " * 400
    ids = [EOT] + tokenizer.encode(text)
    for n in contexts:
        prompt = ids[:n]
        list(gen.stream(prompt, max_new_tokens=new_tokens, temperature=0.0))  # Compile.
        t0 = time.perf_counter()
        it = gen.stream(prompt, max_new_tokens=new_tokens + 1, temperature=0.0)
        next(it)
        t1 = time.perf_counter()
        count = sum(1 for _ in it)
        t2 = time.perf_counter()
        rows.append(dict(context=n, prefill_s=t1 - t0, decode_tok_s=count / (t2 - t1)))
        print(f"context {n:5d}: prefill {t1 - t0:6.2f}s ({n / (t1 - t0):7.0f} tok/s), "
              f"decode {count / (t2 - t1):6.1f} tok/s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="d20_nanochat")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None,
                        help="Compute dtype. Default: the run's")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    dtype = getattr(jnp, args.dtype) if args.dtype else None
    model, tokenizer = generate.load(args.run, str(args.checkpoint_dir.resolve()), args.step, dtype)
    gen = generate.Generator(model)

    print("Greedy completions\n")
    for prompt in PROMPTS:
        out = complete(gen, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=0.0)
        print(f"{prompt}|{out}\n", flush=True)

    print("Samples at temperature 1, top-k 50, from an empty document\n")
    for seed in range(args.num_samples):
        out = tokenizer.decode(list(gen.stream([EOT], max_new_tokens=4 * args.max_tokens,
                                               temperature=1.0, top_k=50, seed=seed,
                                               stop_tokens={EOT})))
        print(f"--- seed {seed}\n{out}\n", flush=True)

    if args.benchmark:
        print("Speed")
        benchmark(gen, tokenizer)


if __name__ == "__main__":
    main()
