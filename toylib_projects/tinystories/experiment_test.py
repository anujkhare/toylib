"""Tests for experiment.py"""

import dataclasses
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, MagicMock, patch

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment


class TestSerializeDataclassConfig:
    """Tests for _serlialize_dataclass_config function."""

    def test_simple_dataclass(self):
        """Test serialization of simple dataclass."""
        config = experiment.TrainingConfig(learning_rate=0.001, max_steps=5000)
        result = experiment._serlialize_dataclass_config(config)

        assert result == {"learning_rate": 0.001, "max_steps": 5000}

    def test_nested_dataclass(self):
        """Test serialization of nested dataclass."""

        @dataclasses.dataclass
        class Inner:
            value: int = 10

        @dataclasses.dataclass
        class Outer:
            inner: Inner = dataclasses.field(default_factory=Inner)
            name: str = "test"

        config = Outer()
        result = experiment._serlialize_dataclass_config(config)

        assert result == {"inner": {"value": 10}, "name": "test"}


class TestExperiment:
    """Tests for Experiment class."""

    def test_config_initialization(self):
        """Test that Experiment initializes properly."""
        mock_dataset = Mock()
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        exp = experiment.Experiment(train_task=train_task)

        assert exp.train_task == train_task
        assert exp.eval_tasks == []
        assert exp.optimizer is not None
        assert exp.opt_state is None
        assert exp.model is None

    @patch("toylib_projects.tinystories.experiment.logger.TensorBoardLogger")
    @patch(
        "toylib_projects.tinystories.experiment.decoder_only_model.DecoderOnlyTransformer"
    )
    @patch("optax.adam")
    def test_init_state(self, mock_opt, mock_model_class, mock_logger):
        """Test that init_state initializes model and optimizer state."""
        mock_dataset = Mock()
        train_task = experiment.Task(name="train", dataset=mock_dataset)
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_opt.return_value.init.return_value = "mock_opt_state"

        exp = experiment.Experiment(train_task=train_task)
        exp.init_state()

        assert exp.model == mock_model
        assert exp.opt_state == "mock_opt_state"
        assert exp.step == 0
        assert exp.train_step_fn is not None
        mock_model_class.assert_called_once()

    @pytest.mark.skip
    def test_e2e(self):
        """Creates a small model and tests saving a checkpoint.

        The file system is mocked out so no actual files are created.
        """

        version = "test_e2e"
        task = experiment.Task(
            name="train",
            dataset=MagicMock(),
        )
        task.dataset.return_value = {
            "inputs": jnp.zeros((1, 10)),
            "targets": jnp.zeros((1, 10)),
        }
        exp = experiment.Experiment(
            model_config=decoder_only_model.ModelConfig(
                vocab_size=5,  # GPT-2 tokenizer vocab size
                num_layers=1,
                qkv_dim=8,
                num_heads=1,
            ),
            training_config=experiment.TrainingConfig(learning_rate=1e-3, max_steps=2),
            checkpoint_config=experiment.CheckpointConfig(
                checkpoint_dir=f"checkpoints/{version}/",
                save_interval_steps=10,
            ),
            train_task=task,
            log_dir=f"tensorboard_logs/{version}/",
        )
        # Initialize state
        exp.init_state()
        # Run outer loop (which includes training and checkpointing)
        exp.outer_loop()
