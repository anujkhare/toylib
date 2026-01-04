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
        bytes_per_token = np.array([1, 2, 3, 4], dtype=np.int32)

        filepath = tmp_path / "bytes_per_token.npy"
        np.save(filepath, bytes_per_token)
        return filepath

    def test_bpb_metric_initialization(self, bytes_per_token_file):
        """Test that BitsPerByte metric loads the bytes per token array."""
        bpb_metric = metrics.BitsPerByte(bytes_per_token_path=str(bytes_per_token_file))

        # Verify the array was loaded
        assert bpb_metric._bytes_per_token is not None
        assert len(bpb_metric._bytes_per_token) == 4
        assert jnp.array_equal(bpb_metric._bytes_per_token, jnp.array([1, 2, 3, 4]))

    def test_bpb_metric_computation(self, bytes_per_token_file):
        """Test that BitsPerByte metric computes bits per byte correctly."""
        bpb_metric = metrics.BitsPerByte(bytes_per_token_path=str(bytes_per_token_file))

        # Create a simple batch with known tokens
        # token_ids: [[0, 1], [2, 3]]
        # bytes: [[1, 2], [3, 4]]
        batch = {
            "inputs": jnp.array([[0, 1], [2, 3]]),
            "targets": jnp.array([[1, 2], [3, 0]]),
            "mask": jnp.ones((2, 2)),
        }

        # Create per-token loss (in nats)
        # Use uniform loss of 1.0 nat for simplicity
        per_token_loss = jnp.ones((2, 2))

        aux = {
            "per_token_loss": per_token_loss,
            "logits": jnp.zeros((2, 2, 4)),  # dummy logits
        }

        # Loss value (unused in BPB calculation)
        loss_val = 1.0

        result = bpb_metric(loss=loss_val, aux=aux, batch=batch)

        # Verify result structure
        assert "bits_per_byte" in result

        # Manual calculation:
        # loss in nats = 1.0 everywhere
        # loss in bits = 1.0 * log2(e) ≈ 1.4427
        # bits_per_byte for each position:
        #   position [0,0]: 1.4427 / 1 = 1.4427
        #   position [0,1]: 1.4427 / 2 = 0.7213
        #   position [1,0]: 1.4427 / 3 = 0.4809
        #   position [1,1]: 1.4427 / 4 = 0.3607
        # mean = (1.4427 + 0.7213 + 0.4809 + 0.3607) / 4 = 0.7514

        expected_bpb = jnp.log2(jnp.e) * jnp.mean(1.0 / jnp.array([1.0, 2.0, 3.0, 4.0]))

        assert jnp.isclose(result["bits_per_byte"], expected_bpb, rtol=1e-5)

    def test_bpb_metric_with_varying_loss(self, bytes_per_token_file):
        """Test BitsPerByte with varying per-token loss values."""
        bpb_metric = metrics.BitsPerByte(bytes_per_token_path=str(bytes_per_token_file))

        # Create batch with tokens: [[0, 1]]
        # bytes: [[1, 2]]
        batch = {
            "inputs": jnp.array([[0, 1]]),
            "targets": jnp.array([[1, 0]]),
            "mask": jnp.ones((1, 2)),
        }

        # Create per-token loss with different values
        # position 0: loss = 2.0 nats
        # position 1: loss = 4.0 nats
        per_token_loss = jnp.array([[2.0, 4.0]])

        aux = {
            "per_token_loss": per_token_loss,
            "logits": jnp.zeros((1, 2, 4)),
        }

        result = bpb_metric(loss=0.0, aux=aux, batch=batch)

        # Manual calculation:
        # position 0: 2.0 * log2(e) / 1 = 2.8854 bits/byte
        # position 1: 4.0 * log2(e) / 2 = 2.8854 bits/byte
        # mean = 2.8854

        log2e = jnp.log2(jnp.e)
        expected_bpb = (2.0 * log2e / 1.0 + 4.0 * log2e / 2.0) / 2.0

        assert jnp.isclose(result["bits_per_byte"], expected_bpb, rtol=1e-5)
