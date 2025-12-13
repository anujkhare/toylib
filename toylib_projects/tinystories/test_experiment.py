"""Tests for experiment.py"""

import dataclasses
from unittest.mock import Mock, MagicMock, patch
from toylib_projects.tinystories import experiment


class TestSerializeDataclassConfig:
    """Tests for _serlialize_dataclass_config function."""

    def test_simple_dataclass(self):
        """Test serialization of simple dataclass."""
        config = experiment.TrainingConfig(
            batch_size=64, learning_rate=0.001, max_steps=5000
        )
        result = experiment._serlialize_dataclass_config(config)

        assert result == {"batch_size": 64, "learning_rate": 0.001, "max_steps": 5000}

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

    @patch("toylib_projects.tinystories.experiment.logger.TensorBoardLogger")
    def test_config_initialization(self, mock_logger):
        """Test that Experiment initializes properly."""
        mock_dataset = Mock()
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        config = experiment.Experiment(train_task=train_task)

        assert config.train_task == train_task
        assert config.eval_tasks == []
        assert config.optimizer is not None
        assert config.opt_state is None
        assert config.model is None
        mock_logger.assert_called_once()

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

        config = experiment.Experiment(train_task=train_task)
        config.init_state()

        assert config.model == mock_model
        assert config.opt_state == "mock_opt_state"
        assert config.step == 0
        assert config.loss_and_grad_fn is not None
        mock_model_class.assert_called_once()
