"""Script to compile and analyze JAX training functions.

This script:
1. Loads an experiment and initializes it
2. Extracts the jitted train_step_fn and eval_step_fn
3. Lowers/compiles them and extracts HLO for analysis
4. Reports peak memory usage
5. Generates Perfetto traces for visualization
6. Supports fake device mesh for simulating data parallel setups

Usage:
    python -m toylib_projects.tinystories.scripts.compile \
        --num_devices 8 \
        --output_dir /tmp/compile_analysis \
        --trace_steps 3

View traces:
    - Perfetto: Open https://ui.perfetto.dev and load the .json.gz trace file
    - xprof/TensorBoard: tensorboard --logdir /tmp/compile_analysis
"""

# IMPORTANT: Setup fake devices BEFORE importing JAX
# JAX reads XLA_FLAGS at import time, so we must set it first
import argparse
import os
import sys


def _setup_fake_devices_from_args():
    """Parse --num_devices from sys.argv and configure XLA before JAX import."""
    # Quick parse just for --num_devices
    for i, arg in enumerate(sys.argv):
        if arg == "--num_devices" and i + 1 < len(sys.argv):
            num_devices = int(sys.argv[i + 1])
            if num_devices > 1:
                os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (
                    f" --xla_force_host_platform_device_count={num_devices}"
                )
            return
        elif arg.startswith("--num_devices="):
            num_devices = int(arg.split("=")[1])
            if num_devices > 1:
                os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (
                    f" --xla_force_host_platform_device_count={num_devices}"
                )
            return


# Set up fake devices before importing JAX
_setup_fake_devices_from_args()

# Now we can import JAX and other modules
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile and analyze JAX training functions"
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of (fake) devices to simulate for data parallelism",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/compile_analysis",
        help="Directory to save analysis outputs (HLO, traces, etc.)",
    )
    parser.add_argument(
        "--trace_steps",
        type=int,
        default=3,
        help="Number of steps to trace for profiling",
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Model depth (number of transformer layers)",
    )
    parser.add_argument(
        "--num_microbatches",
        type=int,
        default=1,
        help="Number of microbatches for gradient accumulation",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--skip_trace",
        action="store_true",
        help="Skip generating Perfetto traces",
    )
    parser.add_argument(
        "--hlo_format",
        type=str,
        choices=["text", "dot", "html", "proto"],
        default="text",
        help="Format for HLO output",
    )
    return parser.parse_args()


def to_mib(bytes_val: int) -> float:
    """Convert bytes to MiB."""
    return bytes_val / (1024**2)


def to_gib(bytes_val: int) -> float:
    """Convert bytes to GiB."""
    return bytes_val / (1024**3)


def analyze_memory(compiled, name: str) -> dict:
    """Analyze memory usage of a compiled function."""
    analysis = compiled.memory_analysis()

    print("\n" + "=" * 60)
    print(f"Memory Analysis: {name}")
    print("=" * 60)
    print(f"  Argument size:      {to_mib(analysis.argument_size_in_bytes):>10.2f} MiB")
    print(f"  Output size:        {to_mib(analysis.output_size_in_bytes):>10.2f} MiB")
    print(f"  Temp/Activations:   {to_mib(analysis.temp_size_in_bytes):>10.2f} MiB")
    print(f"  Alias size:         {to_mib(analysis.alias_size_in_bytes):>10.2f} MiB")
    peak = analysis.argument_size_in_bytes + analysis.temp_size_in_bytes
    print(f"  Peak Memory:        {to_mib(peak):>10.2f} MiB ({to_gib(peak):.2f} GiB)")

    return {
        "argument_size_bytes": analysis.argument_size_in_bytes,
        "output_size_bytes": analysis.output_size_in_bytes,
        "temp_size_bytes": analysis.temp_size_in_bytes,
        "alias_size_bytes": analysis.alias_size_in_bytes,
        "peak_memory_bytes": peak,
    }


def extract_hlo(compiled, name: str, output_dir: Path, fmt: str = "text") -> str:
    """Extract and save HLO from a compiled function."""
    hlo_text = compiled.as_text()

    # Save HLO text
    hlo_path = output_dir / f"{name}_hlo.txt"
    with open(hlo_path, "w") as f:
        f.write(hlo_text)
    print(f"Saved HLO text to: {hlo_path}")

    # Extract cost analysis from HLO if available
    cost_path = output_dir / f"{name}_cost.txt"
    try:
        cost_analysis = compiled.cost_analysis()
        if cost_analysis:
            with open(cost_path, "w") as f:
                for item in cost_analysis:
                    if item:
                        f.write(str(item) + "\n")
            print(f"Saved cost analysis to: {cost_path}")
    except Exception as e:
        print(f"Cost analysis not available: {e}")

    return str(hlo_path)


def analyze_hlo_stats(hlo_text: str, name: str) -> dict:
    """Extract statistics from HLO text."""
    stats = {
        "num_instructions": 0,
        "num_computations": 0,
        "has_custom_calls": False,
        "has_all_reduce": False,
        "has_all_gather": False,
        "has_reduce_scatter": False,
    }

    lines = hlo_text.split("\n")
    for line in lines:
        if line.strip().startswith("ROOT") or "=" in line:
            stats["num_instructions"] += 1
        if line.strip().startswith("ENTRY") or line.strip().startswith("%"):
            stats["num_computations"] += 1
        if "custom-call" in line.lower():
            stats["has_custom_calls"] = True
        if "all-reduce" in line.lower():
            stats["has_all_reduce"] = True
        if "all-gather" in line.lower():
            stats["has_all_gather"] = True
        if "reduce-scatter" in line.lower():
            stats["has_reduce_scatter"] = True

    print("\n" + "=" * 60)
    print(f"HLO Statistics: {name}")
    print("=" * 60)
    print(f"  Total instructions:   {stats['num_instructions']}")
    print(f"  Computations:         {stats['num_computations']}")
    print(f"  Has custom calls:     {stats['has_custom_calls']}")
    print(f"  Has all-reduce:       {stats['has_all_reduce']}")
    print(f"  Has all-gather:       {stats['has_all_gather']}")
    print(f"  Has reduce-scatter:   {stats['has_reduce_scatter']}")

    return stats


def run_profiled_steps(
    exp,
    batch: dict,
    num_steps: int,
    output_dir: Path,
) -> None:
    """Run profiled steps and generate trace."""
    print("\n" + "=" * 60)
    print(f"Running {num_steps} profiled steps...")
    print("=" * 60)

    # Start JAX profiler for Perfetto/TensorBoard
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Use JAX's built-in profiler
    with jax.profiler.trace(str(trace_dir)):
        for step in range(num_steps):
            with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
                exp.model, exp.opt_state, metrics = exp.train_step_fn(
                    exp.model, exp.opt_state, batch
                )
                # Block until computation is done
                jax.block_until_ready(metrics)
            print(
                f"  Step {step + 1}/{num_steps} - loss: {float(metrics.get('loss', 0)):.4f}"
            )

    print(f"\nTrace saved to: {trace_dir}")
    print("  -> Open https://ui.perfetto.dev and load the trace file")
    print(f"  -> Or use: tensorboard --logdir {trace_dir}")


@dataclasses.dataclass
class DummyDataset:
    """A minimal dataset that yields random batches for compilation analysis."""

    batch_size: int
    seq_len: int
    vocab_size: int = 50257
    sharding: object = None  # Optional sharding for device placement

    def __post_init__(self):
        self.tokenizer = None  # Not needed for compilation

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return self.get_batch()

    def get_batch(self) -> dict:
        """Generate a dummy batch, optionally sharded across devices."""
        key = jax.random.PRNGKey(0)
        inputs = jax.random.randint(
            key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        targets = jax.random.randint(
            key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        mask = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.bool_)

        batch = {"inputs": inputs, "targets": targets, "mask": mask}

        if self.sharding is not None:
            batch = {k: jax.device_put(v, self.sharding) for k, v in batch.items()}

        return batch


def create_experiment(
    batch_size_per_device: int,
    seq_len: int,
    depth: int,
    num_microbatches: int,
    vocab_size: int,
    num_devices: int,
):
    """Create an experiment for compilation analysis using the base Experiment class."""
    from toylib_projects.tinystories import experiment
    from toylib_projects.tinystories.train import (
        create_muon_adam_multi_optimizer_config,
        get_model_config,
    )

    # Calculate total batch size
    batch_size = batch_size_per_device * num_devices * num_microbatches

    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  Number of devices:      {num_devices}")
    print(f"  Batch size per device:  {batch_size_per_device}")
    print(f"  Number of microbatches: {num_microbatches}")
    print(f"  Total batch size:       {batch_size}")
    print(f"  Sequence length:        {seq_len}")
    print(f"  Tokens per step:        {batch_size * seq_len}")

    # Model config
    model_config = get_model_config(depth=depth, seq_len=seq_len, vocab_size=vocab_size)

    # Create optimizer config
    optimizer_config = create_muon_adam_multi_optimizer_config(
        muon_lr=1e-4,
        adamw_embed_lr=1e-4,
        adamw_output_lr=1e-4,
    )

    # Create dummy dataset for train task
    dummy_dataset = DummyDataset(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    train_task = experiment.Task(
        name="train",
        dataset=dummy_dataset,
    )

    # Create temp directory for logs/checkpoints
    temp_dir = Path("/tmp/compile_dummy")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create eval task (needed for eval_step_fn to work)
    eval_task = experiment.Task(
        name="eval",
        dataset=dummy_dataset,
    )

    # Create experiment (this sets up sharding, JIT functions, etc.)
    exp = experiment.Experiment(
        model_config=model_config,
        training_config=experiment.TrainingConfig(
            max_steps=1,
            num_microbatches=num_microbatches,
            optimizer_config=optimizer_config,
        ),
        checkpoint_config=experiment.CheckpointConfig(
            checkpoint_dir=str(temp_dir),
        ),
        logger_config=experiment.LoggerConfig(
            log_dir=str(temp_dir),
        ),
        train_task=train_task,
        eval_task=eval_task,
    )

    # Initialize model and optimizer state
    exp.init_state()

    return exp


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("JAX Compilation Analysis")
    print("#" * 60)
    print(f"Output directory: {output_dir}")
    print(f"JAX devices: {jax.local_device_count()} x {jax.devices()[0].platform}")

    # Create experiment using base Experiment class
    exp = create_experiment(
        batch_size_per_device=args.batch_size_per_device,
        seq_len=args.seq_len,
        depth=args.depth,
        num_microbatches=args.num_microbatches,
        vocab_size=args.vocab_size,
        num_devices=args.num_devices,
    )

    # Create dummy batch for compilation (with sharding applied)
    total_batch_size = (
        args.batch_size_per_device * args.num_devices * args.num_microbatches
    )
    dummy_dataset = DummyDataset(
        batch_size=total_batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        sharding=exp.data_sharding,
    )
    batch = dummy_dataset.get_batch()

    # Lower and compile train_step_fn
    print("\n" + "#" * 60)
    print("Compiling train_step_fn...")
    print("#" * 60)

    train_lowered = exp.train_step_fn.lower(exp.model, exp.opt_state, batch)
    train_compiled = train_lowered.compile()

    # Analyze train step
    train_memory = analyze_memory(train_compiled, "train_step_fn")
    train_hlo_path = extract_hlo(
        train_compiled, "train_step", output_dir, args.hlo_format
    )
    train_hlo_text = train_compiled.as_text()
    train_hlo_stats = analyze_hlo_stats(train_hlo_text, "train_step_fn")

    # Lower and compile eval_step_fn
    print("\n" + "#" * 60)
    print("Compiling eval_step_fn...")
    print("#" * 60)

    eval_lowered = exp.eval_step_fn.lower(exp.model, batch)
    eval_compiled = eval_lowered.compile()

    # Analyze eval step
    eval_memory = analyze_memory(eval_compiled, "eval_step_fn")
    eval_hlo_path = extract_hlo(eval_compiled, "eval_step", output_dir, args.hlo_format)
    eval_hlo_text = eval_compiled.as_text()
    eval_hlo_stats = analyze_hlo_stats(eval_hlo_text, "eval_step_fn")

    # Save summary JSON
    summary = {
        "config": {
            "num_devices": args.num_devices,
            "batch_size_per_device": args.batch_size_per_device,
            "total_batch_size": total_batch_size,
            "seq_len": args.seq_len,
            "depth": args.depth,
            "num_microbatches": args.num_microbatches,
            "vocab_size": args.vocab_size,
        },
        "train_step": {
            "memory": train_memory,
            "hlo_stats": train_hlo_stats,
            "hlo_path": train_hlo_path,
        },
        "eval_step": {
            "memory": eval_memory,
            "hlo_stats": eval_hlo_stats,
            "hlo_path": eval_hlo_path,
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary to: {summary_path}")

    # Run profiled steps if not skipped
    if not args.skip_trace:
        run_profiled_steps(exp, batch, args.trace_steps, output_dir)

    # Final summary
    print("\n" + "#" * 60)
    print("Analysis Complete")
    print("#" * 60)
    print("\nOutput files:")
    print(f"  Summary:      {summary_path}")
    print(f"  Train HLO:    {train_hlo_path}")
    print(f"  Eval HLO:     {eval_hlo_path}")
    if not args.skip_trace:
        print(f"  Traces:       {output_dir / 'traces'}")

    print("\n" + "=" * 60)
    print("Peak Memory Summary")
    print("=" * 60)
    print(f"  Train step:   {to_gib(train_memory['peak_memory_bytes']):.2f} GiB")
    print(f"  Eval step:    {to_gib(eval_memory['peak_memory_bytes']):.2f} GiB")

    print("\nVisualization:")
    print(
        f"  Perfetto:     Open https://ui.perfetto.dev and load trace files from {output_dir / 'traces'}"
    )
    print(f"  TensorBoard:  tensorboard --logdir {output_dir / 'traces'}")

    # Cleanup
    exp.cleanup()


if __name__ == "__main__":
    main()
