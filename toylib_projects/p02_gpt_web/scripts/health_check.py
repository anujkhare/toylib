"""Train the first stretch of a run for real, and check that it looks healthy.

Follows the run's full learning-rate schedule but stops early, then judges the logged
curves against what a correctly set up run should show. Prints one JSON line, and exits
with an error if a check fails.

    python -m toylib_projects.p02_gpt_web.scripts.health_check --data-dir /content/data
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np

from toylib import viz
from toylib_projects.p02_gpt_web import experiment, train

FIRST_LOSS_TOLERANCE = 0.1  # Nats from ln(vocab_size), the loss of a uniform guess.
MIN_DROP = 0.3  # Validation bits per byte must fall by at least this share.
RISE_TOLERANCE = 0.01  # Bits per byte that one eval may rise by, for noise.
MAX_TRAIN_VAL_GAP = 0.3  # Nats; a short run on fresh text shouldn't overfit.
UPDATE_TO_WEIGHT_RANGE = (1e-5, 1e-1)  # Outside it, training barely moves or thrashes.
GRAD_NORM_SPIKE = 10.0  # Times the median, after warmup.


def judge(curves: dict, vocab_size: int, warmup_steps: int) -> dict[str, bool]:
    """Return each check's result, from curves as `viz.read_log` returns them."""
    train_steps, train_loss = curves["train/loss"]
    _, val_loss = curves["val/loss"]
    _, bpb = curves["val/bits_per_byte"]
    _, grad_norm = curves["train/grad_norm"]
    _, update_to_weight = curves["train/update_to_weight"]
    after_warmup = [g for s, g in zip(train_steps, grad_norm) if s >= warmup_steps]
    median_grad = sorted(after_warmup)[len(after_warmup) // 2]
    # The first step has no learning rate, so it updates nothing.
    updates = [u for s, u in zip(train_steps, update_to_weight) if s > 0]
    low, high = UPDATE_TO_WEIGHT_RANGE
    values = [*train_loss, *val_loss, *bpb, *grad_norm, *update_to_weight]
    checks = {
        "everything_finite": all(math.isfinite(v) for v in values),
        "starts_at_chance": abs(val_loss[0] - math.log(vocab_size)) < FIRST_LOSS_TOLERANCE,
        "bits_per_byte_falls": all(b <= a + RISE_TOLERANCE for a, b in zip(bpb, bpb[1:])),
        "bits_per_byte_drops_enough": bpb[-1] < (1 - MIN_DROP) * bpb[0],
        "train_matches_val": abs(train_loss[-1] - val_loss[-1]) < MAX_TRAIN_VAL_GAP,
        "updates_in_range": all(low < u < high for u in updates),
        "no_gradient_spikes": max(after_warmup) < GRAD_NORM_SPIKE * median_grad,
    }
    return {name: bool(passed) for name, passed in checks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="d20", choices=sorted(train.RUNS))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="/content/health")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--run-id", default="health")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--microbatches", type=int)
    parser.add_argument("--wandb-project", help="Default: log to a file only")
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()

    run = train.RUNS[args.run]
    if args.microbatch_size and args.microbatches:
        run = dataclasses.replace(
            run,
            num_microbatches=args.microbatches,
            batch_size=args.microbatch_size * args.microbatches,
        )
    run = dataclasses.replace(
        run,
        eval_interval_steps=args.eval_interval,
        save_interval_steps=10 * args.steps,  # Only the start and the end.
    )
    run_dir = pathlib.Path(args.output_dir) / args.run_id
    logger_config = None
    if args.wandb_project:
        logger_config = experiment.WandBLoggerConfig(
            project_name=args.wandb_project,
            user_name=args.wandb_entity,
            log_dir=str(run_dir),
            run_id=args.run_id,
            train_log_interval_steps=5,
            also_log_to_file=True,
        )
    exp = train.create_run_experiment(
        run,
        run_id=args.run_id,
        data_dir=pathlib.Path(args.data_dir),
        output_dir=pathlib.Path(args.output_dir),
        logger_config=logger_config,
    )
    # Stops early, while the schedule built from `run.max_steps` stays the real one.
    exp.training_config.max_steps = args.steps
    exp.outer_loop()
    exp.run_validation()
    exp.cleanup()

    curves = viz.read_log(viz.find_log(run_dir))
    checks = judge(curves, run.vocab_size, run.warmup_steps)
    print(
        json.dumps(
            {
                "checks": checks,
                "val_steps": np.asarray(curves["val/loss"][0]).tolist(),
                "val_loss": np.asarray(curves["val/loss"][1]).tolist(),
                "val_bits_per_byte": np.asarray(curves["val/bits_per_byte"][1]).tolist(),
                "tokens_seen": float(curves["train/tokens_seen"][1][-1]),
                "tokens_per_second": float(curves["train/tokens_per_second"][1][-1]),
            }
        )
    )
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
