"""Where a training step's time goes, on whatever accelerator JAX finds.

Trains for a few real steps and splits each into waiting for data and waiting for the
accelerator, then times evaluation and a checkpoint. Prints one JSON line of results.

    python -m toylib_projects.p02_gpt_web.scripts.benchmark --data-dir /content/data
    python -m toylib_projects.p02_gpt_web.scripts.benchmark --data-dir /content/data --no-remat

MFU uses nanochat's and nanoGPT's definitions, so the numbers compare with theirs.
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from toylib_projects.p02_gpt_web import experiment, metrics, train

UTILIZATION_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw",
    "--format=csv,noheader,nounits",
]


class UtilizationSampler:
    """Sample `nvidia-smi` once a second, while the block inside it runs."""

    def __enter__(self):
        self.rows, self.running = [], True

        def sample():
            while self.running:
                try:
                    line = subprocess.check_output(UTILIZATION_QUERY, text=True)
                    self.rows.append([float(x) for x in line.split(",")])
                except (OSError, subprocess.CalledProcessError, ValueError):
                    return
                time.sleep(1)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        self.thread.join(timeout=5)

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {}
        names = ["gpu_util", "memory_util", "memory_mib", "temperature_c", "power_w"]
        return dict(zip(names, np.round(np.mean(self.rows, axis=0), 1).tolist()))


def run_benchmark(args: argparse.Namespace) -> dict:
    run = dataclasses.replace(
        train.RUNS[args.run],
        seq_len=args.seq_len or train.RUNS[args.run].seq_len,
        num_train_shards=8,
        remat=args.remat,
        num_microbatches=args.microbatches,
        batch_size=args.microbatch_size * args.microbatches,
        dtype=jnp.bfloat16 if args.dtype == "bf16" else jnp.float32,
        attention_impl=args.attention,
    )
    tokens_per_step = run.batch_size * run.seq_len
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        checkpoint_dir = args.checkpoint_dir or str(tmp / "checkpoints")
        start = time.monotonic()
        exp = train.create_run_experiment(
            run,
            run_id="benchmark",
            data_dir=pathlib.Path(args.data_dir),
            output_dir=tmp,
            checkpoint_dir=checkpoint_dir,
            logger_config=experiment.LoggerConfig(
                log_dir=str(tmp), run_id="benchmark", train_log_interval_steps=10
            ),
        )
        batches = iter(exp.train_task.dataset)

        def step() -> tuple[float, float]:
            t0 = time.monotonic()
            batch = next(batches)
            t1 = time.monotonic()
            exp.inner_loop(batch)
            jax.block_until_ready(exp.model)
            exp.step += 1
            return t1 - t0, time.monotonic() - t1

        step()  # Compiles the training step.
        compile_seconds = time.monotonic() - start
        with UtilizationSampler() as sampler:
            times = np.array([step() for _ in range(args.steps)])
        data_seconds, step_seconds = np.median(times, axis=0)

        result = {
            "config": vars(args),
            "device": jax.devices()[0].device_kind,
            "flops_per_token": metrics.flops_per_token(train.model_config(run)),
            "tokens_per_step": tokens_per_step,
            "compile_seconds": round(compile_seconds, 1),
            "data_ms": round(1_000 * float(data_seconds), 2),
            "data_share": round(float(data_seconds / (data_seconds + step_seconds)), 4),
            "step_seconds": round(float(step_seconds), 4),
            "step_seconds_p90": round(float(np.percentile(times[:, 1], 90)), 4),
            "tokens_per_second": round(tokens_per_step / float(data_seconds + step_seconds)),
            "utilization": sampler.summary(),
        }
        peak = metrics.peak_flops()
        if peak:
            achieved = result["tokens_per_second"] * result["flops_per_token"]
            result["mfu"] = round(achieved / peak, 4)

        if args.eval:
            exp.run_validation()  # Compiles the evaluation step.
            start = time.monotonic()
            exp.run_validation()
            result["eval_seconds"] = round(time.monotonic() - start, 2)
            start = time.monotonic()
            exp.sampling_evaluation()
            result["sampling_seconds"] = round(time.monotonic() - start, 2)
        if args.checkpoint:
            start = time.monotonic()
            exp.save_checkpoint()
            exp.ckpt_manager.wait_until_finished()
            result["checkpoint_seconds"] = round(time.monotonic() - start, 2)

        stats = jax.devices()[0].memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            result["peak_gib"] = round(stats["peak_bytes_in_use"] / 2**30, 2)
        losses = train_losses(tmp / "logs_benchmark.txt")
        result["loss_first_last"] = [losses[0], losses[-1]] if losses else None
        exp.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def train_losses(log: pathlib.Path) -> list[float]:
    losses = []
    for line in log.read_text().splitlines():
        if line.startswith("{") and '"train/loss"' in line:
            losses.append(round(json.loads(line)["train/loss"], 3))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="d20", choices=sorted(train.RUNS))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=32)
    parser.add_argument("--seq-len", type=int, help="Default: the run's")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--no-remat", dest="remat", action="store_false")
    parser.add_argument("--attention", choices=["xla", "cudnn"], help="Default: masked matmuls")
    parser.add_argument("--eval", action="store_true", help="Also time an evaluation")
    parser.add_argument("--checkpoint", action="store_true", help="Also time a save")
    parser.add_argument("--checkpoint-dir", help="Local by default; can be gs://")
    print(json.dumps(run_benchmark(parser.parse_args())))


if __name__ == "__main__":
    main()
