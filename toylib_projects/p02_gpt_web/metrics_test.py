"""Tests for metrics.py"""

import jax.numpy as jnp
import numpy as np
import types
import pytest
from pathlib import Path

from toylib_projects.p02_gpt_web import decoder_only_model, metrics


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
            # Total bits over total bytes, not a mean of per-token ratios.
            (
                {
                    "inputs": None,  # unused
                    "targets": jnp.array([[0, 1], [2, 3]]),
                    "mask": jnp.ones((2, 2)),
                },
                # 1 nat per token, and 1 + 2 + 3 + 4 bytes in total.
                jnp.ones((2, 2)),
                4.0 / (np.log(2) * 10.0),
            ),
            # Varying loss per token
            (
                {
                    "inputs": None,  # unused
                    "targets": jnp.array([[0, 1]]),
                    "mask": jnp.ones((1, 2)),
                },
                jnp.array([[2.0, 4.0]]),
                6.0 / (np.log(2) * 3.0),
            ),
            # Tokens masked out
            (
                {
                    "inputs": None,  # unused
                    "targets": jnp.array([[0, 1]]),
                    "mask": jnp.zeros((1, 2)),
                },
                # Varying loss per token
                jnp.array([[2.0, 4.0]]),
                # All tokens should be masked out
                0,
            ),
            # A partly masked batch counts only the valid token
            (
                {
                    "inputs": None,  # unused
                    "targets": jnp.array([[0, 1]]),
                    "mask": jnp.array([[1.0, 0.0]]),
                },
                jnp.array([[2.0, 4.0]]),
                2.0 / np.log(2),
            ),
            # Special tokens in targets count in neither sum
            (
                {
                    "inputs": None,  # unused
                    "targets": jnp.array([[0, 4]]),
                    "mask": jnp.ones((1, 2)),
                },
                # Varying loss per token
                jnp.array([[2.0, 4.0]]),
                2.0 / np.log(2),
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

    def test_uses_targets_not_inputs(self, bytes_per_token_file):
        """The loss at a position is the cost of its target token, so bytes come from there."""
        bpb_metric = metrics.BitsPerByte(bytes_per_token_path=str(bytes_per_token_file))
        batch = {
            "inputs": jnp.array([[3]]),  # 4 bytes
            "targets": jnp.array([[0]]),  # 1 byte
            "mask": jnp.ones((1, 1)),
        }
        result = bpb_metric(None, {"per_token_loss": jnp.ones((1, 1))}, batch)
        assert jnp.isclose(result["bits_per_byte"], 1.0 / np.log(2), rtol=1e-5)


def test_flops_match_nanochat_for_the_d20_shape():
    config = decoder_only_model.ModelConfig(
        qkv_dim=1_280, num_layers=20, num_heads=10, vocab_size=32_768, seq_len=2_048
    )
    assert metrics.flops_per_token(config) == 3_240_099_840


def test_flops_count_windows_and_gates_as_nanochat_does():
    config = decoder_only_model.ModelConfig(
        qkv_dim=768, num_layers=12, num_heads=6, vocab_size=32_768, seq_len=2_048,
        **decoder_only_model.NANOCHAT,
    )
    # Blocks, six value gates of 12 x 6, the output layer and the smear gate.
    matmul_params = 12 * 768**2 * 12 + 6 * 12 * 6 + 768 * 32_768 + 24
    # SSSL tiled over 12 layers: three see 2,048 tokens and nine see 512.
    attention = 12 * 768 * (3 * 2_048 + 9 * 512)
    assert metrics.flops_per_token(config) == 6 * matmul_params + attention


@pytest.mark.parametrize(
    ("kind", "peak"),
    [
        ("NVIDIA A100-SXM4-40GB", 312e12),
        ("NVIDIA H100 80GB HBM3", 989e12),
        ("NVIDIA H100 PCIe", 756e12),
        ("cpu", None),
    ],
)
def test_peak_flops_reads_the_device_name(monkeypatch, kind, peak):
    device = types.SimpleNamespace(device_kind=kind)
    monkeypatch.setattr(metrics.jax, "devices", lambda: [device])
    assert metrics.peak_flops() == peak
