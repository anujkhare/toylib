"""Utilities for JAX compilation analysis.

This module provides:
- DummyDataset: A fake dataset for compilation without real data
- Memory analysis utilities
- HLO extraction and analysis
- Profiled step execution

Usage:
    from toylib_projects.tinystories.scripts.compile import (
        DummyDataset,
        analyze_memory,
        extract_hlo,
        analyze_hlo_stats,
        run_profiled_steps,
        run_compilation_analysis,
    )
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp


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


def run_compilation_analysis(
    exp,
    output_dir: str | Path,
    trace_steps: int = 3,
    skip_trace: bool = False,
    hlo_format: str = "text",
) -> dict:
    """Run full compilation analysis on an experiment.

    Args:
        exp: Initialized Experiment with model and optimizer state
        output_dir: Directory to save analysis outputs
        trace_steps: Number of steps to trace for profiling
        skip_trace: Skip generating Perfetto traces
        hlo_format: Format for HLO output

    Returns:
        Summary dictionary with memory and HLO statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_devices = jax.local_device_count()
    batch_size = exp.train_task.dataset.batch_size
    seq_len = exp.train_task.dataset.seq_len
    vocab_size = exp.train_task.dataset.vocab_size

    print("\n" + "#" * 60)
    print("JAX Compilation Analysis")
    print("#" * 60)
    print(f"Output directory: {output_dir}")
    print(f"JAX devices: {num_devices} x {jax.devices()[0].platform}")

    # Create dummy batch for compilation (with sharding applied)
    dummy_dataset = DummyDataset(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
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
    train_hlo_path = extract_hlo(train_compiled, "train_step", output_dir, hlo_format)
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
    eval_hlo_path = extract_hlo(eval_compiled, "eval_step", output_dir, hlo_format)
    eval_hlo_text = eval_compiled.as_text()
    eval_hlo_stats = analyze_hlo_stats(eval_hlo_text, "eval_step_fn")

    # Save summary JSON
    summary = {
        "config": {
            "platform": jax.devices()[0].platform,
            "num_devices": num_devices,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
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
    if not skip_trace:
        run_profiled_steps(exp, batch, trace_steps, output_dir)

    # Final summary
    print("\n" + "#" * 60)
    print("Analysis Complete")
    print("#" * 60)
    print("\nOutput files:")
    print(f"  Summary:      {summary_path}")
    print(f"  Train HLO:    {train_hlo_path}")
    print(f"  Eval HLO:     {eval_hlo_path}")
    if not skip_trace:
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

    return summary
