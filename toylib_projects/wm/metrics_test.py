"""Tests for metrics.py.

Forked from `toylib_projects/tinystories/metrics_test.py` keeping only the
``TestLossMetric`` class. The text-domain ``TestBitsPerByteMetric`` is not
forked because we don't carry ``BitsPerByte`` in this project (see
``metrics.py`` docstring).
"""

import jax.numpy as jnp

from toylib_projects.wm import metrics


class TestLossMetric:
    """Tests for the Loss pass-through metric."""

    def test_loss_metric_returns_loss(self) -> None:
        """Loss metric returns the loss value under the 'loss' key."""
        loss_metric = metrics.Loss()
        loss_val = 2.5
        aux = {"logits": jnp.array([1.0, 2.0])}
        batch = {"inputs": jnp.array([0, 1])}

        result = loss_metric(loss=loss_val, aux=aux, batch=batch)

        assert "loss" in result
        assert result["loss"] == 2.5

    def test_loss_metric_ignores_aux_and_batch(self) -> None:
        """Loss is independent of aux / batch contents."""
        loss_metric = metrics.Loss()

        r1 = loss_metric(loss=1.0, aux=None, batch=None)
        r2 = loss_metric(loss=1.0, aux={"anything": 42}, batch={"x": jnp.zeros((4,))})

        assert r1 == r2

    def test_loss_metric_passes_through_jax_arrays(self) -> None:
        """Loss values that are jnp arrays survive untouched (no host transfer)."""
        loss_metric = metrics.Loss()
        loss_val = jnp.array(0.7)
        result = loss_metric(loss=loss_val, aux={}, batch=None)

        assert result["loss"] is loss_val  # identity, not just equality
