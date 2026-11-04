"""Tests for health_check.py."""

import math

import pytest

from toylib_projects.p02_gpt_web.scripts import health_check

VOCAB = 32_768
FIRST = math.log(VOCAB)


def curves(**overrides):
    steps = [0, 50, 100]
    result = {
        "train/loss": (steps, [FIRST, 6.0, 5.0]),
        "val/loss": (steps, [FIRST, 6.1, 5.1]),
        "val/bits_per_byte": (steps, [3.2, 1.9, 1.6]),
        "train/grad_norm": (steps, [1.0, 1.2, 1.1]),
        "train/update_to_weight": (steps, [1e-3, 2e-3, 2e-3]),
    }
    for name, values in overrides.items():
        result[name.replace("__", "/")] = (steps, values)
    return result


def test_a_healthy_run_passes():
    assert all(health_check.judge(curves(), VOCAB, warmup_steps=40).values())


@pytest.mark.parametrize(
    ("override", "check"),
    [
        ({"train__loss": [FIRST, math.nan, 5.0]}, "everything_finite"),
        ({"val__loss": [9.0, 6.1, 5.1]}, "starts_at_chance"),
        ({"val__bits_per_byte": [3.2, 1.6, 1.9]}, "bits_per_byte_falls"),
        ({"val__bits_per_byte": [3.2, 3.0, 2.9]}, "bits_per_byte_drops_enough"),
        ({"val__loss": [FIRST, 6.1, 5.9]}, "train_matches_val"),
        ({"train__update_to_weight": [1e-3, 2e-7, 2e-3]}, "updates_in_range"),
    ],
)
def test_a_problem_fails_its_check(override, check):
    assert not health_check.judge(curves(**override), VOCAB, warmup_steps=40)[check]


def test_a_gradient_spike_fails():
    steps = list(range(0, 100, 10))
    values = curves()
    values["train/grad_norm"] = (steps, [1.0] * 9 + [50.0])
    values["train/loss"] = (steps, [FIRST] + [5.0] * 9)
    assert not health_check.judge(values, VOCAB, warmup_steps=0)["no_gradient_spikes"]
