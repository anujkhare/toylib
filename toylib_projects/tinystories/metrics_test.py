"""Tests for metrics.py"""

import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from toylib_projects.tinystories import metrics


class TestLossMetric:
    """Tests for Loss metric."""

    def test_loss_metric_returns_loss(self):
        """Test that Loss metric returns the loss value."""
        loss_metric = metrics.Loss()
        loss_val = 2.5
        aux = {"logits": jnp.array([1.0, 2.0])}
        batch = {"inputs": jnp.array([0, 1])}

        result = loss_metric(loss=loss_val, aux=aux, batch=batch)

        assert "loss" in result
        assert result["loss"] == 2.5


class TestBitsPerByteMetric:
    """Tests for BitsPerByte metric."""

    @pytest.fixture
    def bytes_per_token_file(self, tmp_path: Path) -> Path:
        """Create a temporary bytes per token file."""
        # Create a simple mapping: token_id -> bytes_per_token
        # For testing, use a small vocab where:
        # token 0 = 1 byte, token 1 = 2 bytes, token 2 = 3 bytes, token 3 = 4 bytes
        # token 4 is a special token and should be excluded (set to -1 bytes)
        bytes_per_token = np.array([1, 2, 3, 4, -1], dtype=np.int32)

        filepath = tmp_path / "bytes_per_token.npy"
        np.save(filepath, bytes_per_token)
        return filepath

    @pytest.mark.parametrize(
        "batch,per_token_loss,expected_bpb",
        [
            (
                {
                    "inputs": jnp.array([[0, 1], [2, 3]]),
                    "mask": jnp.ones((2, 2)),
                    "targets": None,  # unused
                },
                # Use uniform loss of 1.0 nat for simplicity
                jnp.ones((2, 2)),
                # Manual calculation:
                # loss in nats = 1.0 everywhere
                # loss in bits = 1.0 * log2(e) ≈ 1.4427
                # bits_per_byte for each position:
                #   position [0,0]: 1.4427 / 1 = 1.4427
                #   position [0,1]: 1.4427 / 2 = 0.7213
                #   position [1,0]: 1.4427 / 3 = 0.4809
                #   position [1,1]: 1.4427 / 4 = 0.3607
                # mean = (1.4427 + 0.7213 + 0.4809 + 0.3607) / 4 = 0.7514
                jnp.log2(jnp.e) * jnp.mean(1.0 / jnp.array([1.0, 2.0, 3.0, 4.0])),
            ),
            # Varying loss per token
            (
                {
                    "inputs": jnp.array([[0, 1]]),
                    "targets": None,  # unused
                    "mask": jnp.ones((1, 2)),
                },
                jnp.array([[2.0, 4.0]]),
                # Manual calculation:
                # position 0: 2.0 * log2(e) / 1 = 2.8854 bits/byte
                # position 1: 4.0 * log2(e) / 2 = 2.8854 bits/byte
                # mean = 2.8854
                jnp.log2(jnp.e) * (2.0 / 1.0 + 4.0 / 2.0) / 2.0,
            ),
            # Tokens masked out
            (
                {
                    "inputs": jnp.array([[0, 1]]),
                    "targets": None,  # unused
                    "mask": jnp.zeros((1, 2)),
                },
                # Varying loss per token
                jnp.array([[2.0, 4.0]]),
                # All tokens should be masked out
                0,
            ),
            # Special tokens in inputs
            (
                {
                    "inputs": jnp.array([[0, 5]]),
                    "targets": None,  # unused
                    "mask": jnp.ones((1, 2)),
                },
                # Varying loss per token
                jnp.array([[2.0, 4.0]]),
                # All tokens should be masked out
                jnp.log2(jnp.e) * (2.0 / 1.0),
            ),
        ],
    )
    def test_bpb_metric_computation(
        self,
        batch,
        per_token_loss,
        expected_bpb,
        bytes_per_token_file,
    ):
        """Test that BitsPerByte metric computes bits per byte correctly."""
        bpb_metric = metrics.BitsPerByte(bytes_per_token_path=str(bytes_per_token_file))

        aux = {
            "per_token_loss": per_token_loss,
            "logits": None,  # unused
        }

        # Loss value - unused
        loss_val = None

        result = bpb_metric(loss=loss_val, aux=aux, batch=batch)

        # Verify result structure
        assert "bits_per_byte" in result

        assert jnp.isclose(result["bits_per_byte"], expected_bpb, rtol=1e-5)
