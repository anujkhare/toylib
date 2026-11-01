"""
Evaluation script for the TinyStories JAX model trained in this repo.

Loads trained models from orbax checkpoints and supports three eval modes:
  --eval bpb    : Bits per byte on the validation split
  --eval sample : Generate text samples from the model
  --eval core   : CORE metric (accuracy on ICL tasks)

Default: --eval core,bpb,sample
"""

import csv
import json
import os
import random
import time
import yaml
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import transformers

import toylib_projects.tinystories.data as ts_data
import toylib_projects.tinystories.decoder_only_model as ts_model
import toylib_projects.tinystories.metrics as ts_metrics
import toylib_projects.tinystories.train as ts_train
import toylib_projects.tinystories.tokenizer.bytes_per_token as ts_bpt

EVAL_BUNDLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "eval_bundle",
)

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
]


# -----------------------------------------------------------------------------
# Checkpoint loading


def load_model_from_checkpoint(
    checkpoint_dir: str,
    step: int | None,
    depth: int,
    seq_len: int,
    vocab_size: int = 50257,
) -> tuple[ts_model.DecoderOnlyTransformer, int, ts_model.ModelConfig]:
    """Load model weights from an orbax checkpoint.

    The model config (depth, seq_len, vocab_size) must match the training run.
    """
    model_config = ts_train.get_model_config(depth=depth, seq_len=seq_len, vocab_size=vocab_size)

    # Build a template with the right pytree structure so orbax can restore into it.
    template = ts_model.DecoderOnlyTransformer(
        config=model_config, key=jax.random.key(0)
    )
    template.init()
    template_np = jax.tree.map(np.asarray, template)

    ckpt_manager = ocp.CheckpointManager(checkpoint_dir)
    if step is None:
        step = ckpt_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    restored = ckpt_manager.restore(
        step,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(template_np)),
    )
    ckpt_manager.close()

    model = jax.tree.map(jnp.asarray, restored["model"])
    print(f"Loaded checkpoint at step {step} from {checkpoint_dir}")
    return model, step, model_config


# -----------------------------------------------------------------------------
# Sampling


def run_sampling(
    model: ts_model.DecoderOnlyTransformer,
    model_config: ts_model.ModelConfig,
    tokenizer,
    max_tokens: int = 64,
) -> None:
    print("\n" + "=" * 70)
    print("Samples")
    print("=" * 70)

    seq_len = model_config.seq_len
    # max_output_tokens must be static since lax.scan uses it as loop length.
    generate = jax.jit(
        ts_model.sample, static_argnames=["max_output_tokens", "top_k"]
    )

    for prompt in DEFAULT_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        padded = np.zeros(seq_len, dtype=np.int32)
        padded[: len(prompt_ids)] = prompt_ids
        generated = generate(
            model=model,
            input_tokens=jnp.array(padded),
            prompt_len=len(prompt_ids),
            key=jax.random.key(42),
            max_output_tokens=max_tokens,
            temperature=1.0,
            top_k=5,
        )
        output = tokenizer.decode(generated.tolist(), skip_special_tokens=True)
        print(f"\nPrompt : {prompt}")
        print(f"Output : {output}")


# -----------------------------------------------------------------------------
# BPB evaluation


def run_bpb(
    model: ts_model.DecoderOnlyTransformer,
    model_config: ts_model.ModelConfig,
    dataset_path: str,
    val_split: str,
    batch_size: int,
    num_batches: int = 20,
) -> float:
    print("\n" + "=" * 70)
    print("Bits Per Byte Evaluation")
    print("=" * 70)

    bpt_path = "/tmp/bpt_gpt2.npy"
    if not os.path.exists(bpt_path):
        bpt_arr = ts_bpt.compute_bytes_per_token(tokenizer_name="gpt2")
        np.save(bpt_path, bpt_arr)

    val_dataset = ts_data.BatchedTokenizedDatasetGrain(
        dataset_path=dataset_path,
        split=val_split,
        batch_size=batch_size,
        seq_len=model_config.seq_len,
    )

    bpb_metric = ts_metrics.BitsPerByte(bpt_path)
    forward_jit = jax.jit(
        lambda m, batch: ts_model.train_step(m, batch, return_aux=True)
    )

    total_bpb = 0.0
    count = 0
    for batch in val_dataset:
        _, aux = forward_jit(model, batch)
        result = bpb_metric(loss=None, aux=aux, batch=batch)
        total_bpb += float(result["bits_per_byte"])
        count += 1
        if count >= num_batches:
            break

    avg_bpb = total_bpb / count if count > 0 else float("nan")
    print(f"Val bits per byte ({count} batches): {avg_bpb:.6f}")
    return avg_bpb


# -----------------------------------------------------------------------------
# CORE evaluation


def _sum_continuation_logprob(
    log_probs: jax.Array,  # [seq_len, vocab_size]
    prefix_len: int,
    cont_ids: list[int],
    seq_len: int,
) -> float:
    """Sum log-probs of continuation tokens given the prefix length."""
    total = 0.0
    for j, token_id in enumerate(cont_ids):
        pos = prefix_len - 1 + j  # logits at pos predict token at pos+1
        if pos >= seq_len - 1:
            break
        total += float(log_probs[pos, token_id])
    return total


def _encode_pair(
    tokenizer, prefix: str, continuation: str, seq_len: int
) -> tuple[np.ndarray, int, list[int]]:
    """Tokenize prefix+continuation into a padded token array.

    Returns (tokens [seq_len], prefix_len, cont_ids).
    prefix_len is the effective start of the continuation in the padded array.
    """
    prefix_ids = tokenizer.encode(prefix)
    cont_ids = tokenizer.encode(continuation)
    full_ids = prefix_ids + cont_ids
    if len(full_ids) > seq_len:
        # Drop from the front of the prefix, keeping the continuation intact.
        full_ids = full_ids[-seq_len:]
        prefix_len = len(full_ids) - len(cont_ids)
    else:
        prefix_len = len(prefix_ids)
    tokens = np.zeros(seq_len, dtype=np.int32)
    tokens[: len(full_ids)] = full_ids
    return tokens, prefix_len, cont_ids


def _build_fewshot_prefix(
    data: list[dict], idx: int, num_fewshot: int, task_type: str, cont_delim: str
) -> str:
    """Collect few-shot examples from data, skipping index idx."""
    parts = []
    for j, ex in enumerate(data):
        if j == idx or len(parts) >= num_fewshot:
            continue
        if task_type == "multiple_choice":
            parts.append(ex["query"] + cont_delim + ex["choices"][ex["gold"]])
        elif task_type == "language_modeling":
            parts.append(ex["context"] + cont_delim + ex["continuation"])
        elif task_type == "schema":
            parts.append(ex["context_options"][ex["gold"]] + " " + ex["continuation"])
    return ("\n\n".join(parts) + "\n\n") if parts else ""


def _evaluate_task(forward_jit, tokenizer, data, task_meta, seq_len) -> float:
    """Evaluate a single ICL task, returning accuracy."""
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    cont_delim = task_meta["continuation_delimiter"]
    correct = 0

    for idx, example in enumerate(data):
        prefix_base = _build_fewshot_prefix(data, idx, num_fewshot, task_type, cont_delim)

        if task_type == "multiple_choice":
            query = prefix_base + example["query"] + cont_delim
            scores = []
            for choice in example["choices"]:
                tokens, prefix_len, cont_ids = _encode_pair(
                    tokenizer, query, choice, seq_len
                )
                logits = forward_jit(jnp.array(tokens))
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                scores.append(
                    _sum_continuation_logprob(log_probs, prefix_len, cont_ids, seq_len)
                )
            correct += int(np.argmax(scores) == example["gold"])

        elif task_type == "language_modeling":
            context = prefix_base + example["context"] + cont_delim
            ctx_ids = tokenizer.encode(context)
            cont_first = tokenizer.encode(example["continuation"])[0]
            if len(ctx_ids) >= seq_len:
                ctx_ids = ctx_ids[-(seq_len - 1) :]
            tokens = np.zeros(seq_len, dtype=np.int32)
            tokens[: len(ctx_ids)] = ctx_ids
            logits = forward_jit(jnp.array(tokens))
            pred = int(jnp.argmax(logits[len(ctx_ids) - 1]))
            correct += int(pred == cont_first)

        elif task_type == "schema":
            cont = " " + example["continuation"]
            cont_ids = tokenizer.encode(cont)
            scores = []
            for ctx_opt in example["context_options"]:
                tokens, prefix_len, _ = _encode_pair(
                    tokenizer, prefix_base + ctx_opt, cont, seq_len
                )
                logits = forward_jit(jnp.array(tokens))
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                scores.append(
                    _sum_continuation_logprob(log_probs, prefix_len, cont_ids, seq_len)
                )
            correct += int(np.argmax(scores) == example["gold"])

    return correct / len(data) if data else 0.0


def run_core_eval(
    model: ts_model.DecoderOnlyTransformer,
    model_config: ts_model.ModelConfig,
    tokenizer,
    max_per_task: int = -1,
) -> dict:
    print("\n" + "=" * 70)
    print("CORE Evaluation")
    print("=" * 70)

    config_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    data_base = os.path.join(EVAL_BUNDLE_DIR, "eval_data")
    meta_path = os.path.join(EVAL_BUNDLE_DIR, "eval_meta_data.csv")

    with open(config_path) as f:
        tasks = yaml.safe_load(f)["icl_tasks"]

    random_baselines = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    # JIT the forward pass with the model closed over (single set of weights).
    forward_jit = jax.jit(lambda tokens: model(tokens))
    seq_len = model_config.seq_len

    results = {}
    centered_results = {}

    for task in tasks:
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = os.path.join(data_base, task["dataset_uri"])
        with open(data_path) as f:
            data = [json.loads(line) for line in f]

        rng = random.Random(1337)
        rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        t0 = time.time()
        print(
            f"  {label} ({task_meta['num_fewshot']}-shot {task_meta['task_type']}, "
            f"n={len(data)})...",
            end="",
            flush=True,
        )
        acc = _evaluate_task(forward_jit, tokenizer, data, task_meta, seq_len)
        rb = random_baselines.get(label, 0.0)
        centered = (acc - 0.01 * rb) / (1.0 - 0.01 * rb)
        results[label] = acc
        centered_results[label] = centered
        print(f" acc={acc:.4f} centered={centered:.4f} ({time.time() - t0:.1f}s)")

    core_metric = sum(centered_results.values()) / len(centered_results)
    print(f"\nCORE metric: {core_metric:.4f}")
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TinyStories JAX model")
    parser.add_argument(
        "--eval",
        type=str,
        default="core,bpb,sample",
        help="Comma-separated eval modes: core, bpb, sample (default: all)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Orbax checkpoint directory (e.g. /tmp/checkpoints/<run_id>)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Model depth used during training (default: 12)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length used during training (default: 2048)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size (default: 50257 for GPT-2)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to parquet dataset (required for --eval bpb)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Validation split name (default: val)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for BPB evaluation (default: 8)",
    )
    parser.add_argument(
        "--bpb-batches",
        type=int,
        default=20,
        help="Number of batches to evaluate for BPB (default: 20)",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Max examples per CORE task (-1 = all; use ~100 for a quick run)",
    )
    parser.add_argument(
        "--max-sample-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per sample prompt (default: 64)",
    )
    args = parser.parse_args()

    eval_modes = {m.strip() for m in args.eval.split(",")}
    invalid = eval_modes - {"core", "bpb", "sample"}
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: core, bpb, sample")

    if "bpb" in eval_modes and args.dataset_path is None:
        parser.error("--dataset-path is required for --eval bpb")

    model, step, model_config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        depth=args.depth,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    if "sample" in eval_modes:
        run_sampling(model, model_config, tokenizer, max_tokens=args.max_sample_tokens)

    if "bpb" in eval_modes:
        run_bpb(
            model,
            model_config,
            args.dataset_path,
            args.val_split,
            args.batch_size,
            num_batches=args.bpb_batches,
        )

    if "core" in eval_modes:
        core_results = run_core_eval(
            model, model_config, tokenizer, max_per_task=args.max_per_task
        )
        output_dir = os.path.join(args.checkpoint_dir, "eval")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"core_eval_step{step:06d}.csv")
        with open(output_csv, "w", newline="") as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in core_results["results"]:
                acc = core_results["results"][label]
                centered = core_results["centered_results"][label]
                f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
            f.write(
                f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n"
            )
        print(f"Results written to {output_csv}")


if __name__ == "__main__":
    main()
