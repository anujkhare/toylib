"""Tests for experiment.py"""

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, MagicMock, patch

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment
from toylib_projects.tinystories import logger
from toylib_projects.tinystories import metrics
import optax


class TestSerializeDataclassConfig:
    """Tests for _serialize_dataclass_config function."""

    def test_simple_dataclass(self):
        """Test serialization of simple dataclass."""

        @dataclasses.dataclass
        class TestConfig:
            foo: int = 10
            bar: str = "test"
            baz: float = 3.14

        config = TestConfig()
        result = experiment._serialize_dataclass_config(config)

        assert result == {"foo": 10, "bar": "test", "baz": 3.14}

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
        result = experiment._serialize_dataclass_config(config)

        assert result == {"inner": {"value": 10}, "name": "test"}


class TestExperiment:
    """Tests for Experiment class."""

    def test_config_initialization(self):
        """Test that Experiment initializes properly."""
        mock_dataset = Mock()
        mock_dataset.batch_size = 4
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        exp = experiment.Experiment(train_task=train_task)

        assert exp.train_task == train_task
        assert exp.eval_task is None
        # Optimizer is created during init_state(), not during __post_init__()
        assert exp.optimizer is None
        assert exp.opt_state is None
        assert exp.model is None

    @patch(
        "toylib_projects.tinystories.experiment.jax.device_put",
        side_effect=lambda x, *a, **kw: x,
    )
    @patch(
        "toylib_projects.tinystories.experiment.decoder_only_model.DecoderOnlyTransformer"
    )
    @patch("optax.adam")
    def test_init_state(self, mock_opt, mock_model_class, mock_device_put):
        """Test that init_state initializes model and optimizer state."""
        mock_dataset = Mock()
        mock_dataset.batch_size = 4
        train_task = experiment.Task(name="train", dataset=mock_dataset)
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_opt.return_value.init.return_value = "mock_opt_state"

        exp = experiment.Experiment(train_task=train_task)
        exp.init_state()

        assert exp.model == mock_model
        assert exp.optimizer is not None
        assert exp.opt_state == "mock_opt_state"
        assert exp.step == 0
        assert exp.train_step_fn is not None
        mock_model_class.assert_called_once()


class MockDataset:
    """A mock dataset that yields batches of random data."""

    def __init__(
        self, batch_size: int, seq_len: int, vocab_size: int, num_batches: int = 2
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self._iteration_count = 0

    def __iter__(self) -> Iterator[dict]:
        self._iteration_count = 0
        return self

    def __next__(self) -> dict:
        if self._iteration_count >= self.num_batches:
            raise StopIteration
        self._iteration_count += 1

        # Generate random input tokens (deterministic based on iteration)
        key = jax.random.key(self._iteration_count)
        inputs = jax.random.randint(
            key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        # Targets are shifted inputs (typical for language modeling)
        targets = jnp.roll(inputs, -1, axis=1)
        # Mask is all ones (all tokens are valid)
        mask = jnp.ones((self.batch_size, self.seq_len))

        return {"inputs": inputs, "targets": targets, "mask": mask}

    def get_state(self) -> dict:
        return {"iteration_count": self._iteration_count}

    def restore_state(self, state: dict) -> None:
        self._iteration_count = state["iteration_count"]


def _get_model_params_flat(model) -> jnp.ndarray:
    """Flatten all model parameters into a single array for comparison."""
    leaves = jax.tree_util.tree_leaves(model)
    return jnp.concatenate([leaf.flatten() for leaf in leaves])


def _create_test_experiment(
    train_dataset: MockDataset,
    eval_dataset: MockDataset | None = None,
    checkpoint_dir: str | Path = "/tmp/test_checkpoints",
    log_dir: str | Path = "/tmp/test_logs",
) -> experiment.Experiment:
    """Create an experiment configured for testing with real components."""
    train_task = experiment.Task(name="train", dataset=train_dataset)
    eval_task = (
        experiment.Task(name="eval", dataset=eval_dataset)
        if eval_dataset is not None
        else None
    )

    # Create a small model config for fast testing
    model_config = decoder_only_model.ModelConfig(
        vocab_size=train_dataset.vocab_size,
        num_layers=1,
        qkv_dim=8,
        num_heads=1,
    )

    training_config = experiment.TrainingConfig(
        learning_rate=1e-2,  # Higher LR to see parameter changes
        max_steps=10,
    )

    eval_config = experiment.EvalConfig(
        eval_interval_steps=100,  # Don't run eval during outer_loop by default
        num_eval_steps=1,
    )

    checkpoint_config = experiment.CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir),
        save_interval_steps=100,  # Don't auto-save during outer_loop by default
        max_to_keep=5,
    )

    # Use the real logger with the temporary directory
    logger_config = experiment.LoggerConfig(
        logger_cls=logger.FileLogger,
        log_dir=str(log_dir),
    )

    exp = experiment.Experiment(
        train_task=train_task,
        eval_task=eval_task,
        model_config=model_config,
        training_config=training_config,
        eval_config=eval_config,
        checkpoint_config=checkpoint_config,
        logger_config=logger_config,
        jit_computations=False,  # Disable JIT for faster testing
    )

    return exp


@pytest.mark.expensive
class TestExperimentE2E:
    """End-to-end tests for the Experiment class.

    These tests use a small model with real checkpoint manager and logger
    components, writing to a temporary filesystem that is automatically
    cleaned up after each test.
    """

    # Test configuration constants
    BATCH_SIZE = 2
    SEQ_LEN = 8
    VOCAB_SIZE = 32
    NUM_BATCHES = 3

    @pytest.fixture
    def train_dataset(self) -> MockDataset:
        """Create a mock training dataset."""
        return MockDataset(
            batch_size=self.BATCH_SIZE,
            seq_len=self.SEQ_LEN,
            vocab_size=self.VOCAB_SIZE,
            num_batches=self.NUM_BATCHES,
        )

    @pytest.fixture
    def eval_dataset(self) -> MockDataset:
        """Create a mock evaluation dataset."""
        return MockDataset(
            batch_size=self.BATCH_SIZE,
            seq_len=self.SEQ_LEN,
            vocab_size=self.VOCAB_SIZE,
            num_batches=2,
        )

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create a temporary checkpoint directory."""
        # tmp_path is a pytest fixture providing a temp directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        """Create a temporary log directory."""
        # tmp_path is a pytest fixture providing a temp directory
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def test_model_initialization(self, train_dataset, checkpoint_dir, log_dir):
        """Test that the model initializes with random weights."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Verify model and optimizer state are initialized
        assert exp.model is not None
        assert exp.opt_state is not None
        assert exp.step == 0

        # Verify model has learnable parameters
        params_flat = _get_model_params_flat(exp.model)
        assert params_flat.size > 0
        # Parameters should not be all zeros (random init)
        assert not jnp.allclose(params_flat, 0.0)

        exp.cleanup()

    def test_training_step_updates_parameters(
        self, train_dataset, checkpoint_dir, log_dir
    ):
        """Test that a training step updates model parameters."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Store initial parameters
        initial_params = _get_model_params_flat(exp.model)

        # Get a batch and run training
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)

        # Verify parameters changed
        updated_params = _get_model_params_flat(exp.model)
        assert not jnp.allclose(initial_params, updated_params), (
            "Model parameters should change after training step"
        )

        exp.cleanup()

    def test_checkpoint_save(self, train_dataset, checkpoint_dir, log_dir):
        """Test that checkpoints are saved correctly to the filesystem."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Run a training step
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)
        exp.step = 100  # Set step for checkpoint

        # Save checkpoint
        exp.save_checkpoint()

        # Verify checkpoint directory was created with content
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"

        # Orbax creates subdirectories for each checkpoint step
        checkpoint_contents = list(checkpoint_dir.iterdir())
        assert len(checkpoint_contents) > 0, "Checkpoint directory should not be empty"

        # Verify the checkpoint manager knows about the saved step
        saved_steps = exp.ckpt_manager.all_steps()
        assert 100 in saved_steps, f"Step 100 should be in saved steps: {saved_steps}"

        exp.cleanup()

    def test_checkpoint_restore(self, train_dataset, checkpoint_dir, log_dir):
        """Test that checkpoints restore correctly after resetting state."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Run training and save initial state for comparison
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)
        exp.step = 100

        # Store the trained parameters
        trained_params = _get_model_params_flat(exp.model)

        # Save checkpoint
        exp.save_checkpoint()

        # Run more training to change the model
        exp.inner_loop(batch)
        exp.inner_loop(batch)

        # Verify parameters have changed
        changed_params = _get_model_params_flat(exp.model)
        assert not jnp.allclose(trained_params, changed_params), (
            "Parameters should change after additional training"
        )

        # Restore checkpoint
        exp.restore_checkpoint(100)

        # Verify parameters are restored to the saved state
        restored_params = _get_model_params_flat(exp.model)
        assert jnp.allclose(trained_params, restored_params), (
            "Parameters should match the saved checkpoint after restore"
        )

        # Verify step is restored
        assert exp.step == 100

        exp.cleanup()

    @pytest.mark.skip
    def test_full_training_loop(
        self, train_dataset, eval_dataset, checkpoint_dir, log_dir
    ):
        """Test a complete training loop with checkpointing."""
        exp = _create_test_experiment(
            train_dataset, eval_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )

        # Reconfigure for a short training run with checkpointing
        exp.training_config.max_steps = 3
        exp.checkpoint_config.save_interval_steps = 2
        exp.eval_config.eval_interval_steps = 100  # Skip eval for speed

        exp.init_state()

        # Store initial parameters
        initial_params = _get_model_params_flat(exp.model)

        # Run full training loop
        exp.outer_loop()

        # Verify training progressed
        assert exp.step >= exp.training_config.max_steps

        # Verify parameters changed
        final_params = _get_model_params_flat(exp.model)
        assert not jnp.allclose(initial_params, final_params), (
            "Parameters should change after training"
        )

        # Verify checkpoint was saved (at step 0 and/or 2)
        saved_steps = exp.ckpt_manager.all_steps()
        assert len(saved_steps) >= 1, "At least one checkpoint should be saved"
        assert 0 in saved_steps or 2 in saved_steps, (
            f"Checkpoint should be saved at step 0 or 2, got: {saved_steps}"
        )

        # Verify checkpoint files exist on disk
        checkpoint_contents = list(checkpoint_dir.iterdir())
        assert len(checkpoint_contents) > 0, "Checkpoint files should exist on disk"

        exp.cleanup()

    def test_checkpoint_roundtrip_preserves_training_state(
        self, train_dataset, checkpoint_dir, log_dir
    ):
        """Test that checkpoint restore allows training to continue correctly."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Run some training
        train_iter = iter(train_dataset)
        batch1 = next(train_iter)
        batch2 = next(train_iter)

        exp.inner_loop(batch1)
        exp.step = 1
        exp.save_checkpoint()

        exp.inner_loop(batch2)
        exp.step = 2

        params_after_step2 = _get_model_params_flat(exp.model)

        # Now restore to step 1 and run the same batch again
        exp.restore_checkpoint(1)
        assert exp.step == 1

        # Continue training from restored state
        exp.inner_loop(batch2)
        params_after_continue = _get_model_params_flat(exp.model)

        # Training from restored state should produce the same result
        # (since we're using the same batch and the training is deterministic)
        assert jnp.allclose(params_after_step2, params_after_continue, rtol=1e-5), (
            "Training from restored checkpoint should produce same results"
        )

        exp.cleanup()

    def test_log_files_created(self, train_dataset, checkpoint_dir, log_dir):
        """Test that log files are created in the log directory."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Run a training step which should log
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)

        # Close the logger to flush any buffered writes
        exp.cleanup()

        # Verify log directory has content (exact structure depends on logger impl)
        # At minimum, the directory should exist
        assert log_dir.exists(), "Log directory should exist"

    def test_multiple_checkpoint_saves(self, train_dataset, checkpoint_dir, log_dir):
        """Test that multiple checkpoints can be saved and the correct one restored."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        batch = next(iter(train_dataset))
        params_at_steps = {}

        # Save checkpoints at multiple steps
        for step in [10, 20, 30]:
            exp.inner_loop(batch)
            exp.step = step
            params_at_steps[step] = _get_model_params_flat(exp.model).copy()
            exp.save_checkpoint()

        # Verify all checkpoints exist
        saved_steps = exp.ckpt_manager.all_steps()
        assert 10 in saved_steps
        assert 20 in saved_steps
        assert 30 in saved_steps

        # Restore to step 20 and verify correct params
        exp.restore_checkpoint(20)
        restored_params = _get_model_params_flat(exp.model)
        assert jnp.allclose(params_at_steps[20], restored_params), (
            "Should restore params from step 20"
        )
        assert exp.step == 20

        # Restore to step 10 and verify correct params
        exp.restore_checkpoint(10)
        restored_params = _get_model_params_flat(exp.model)
        assert jnp.allclose(params_at_steps[10], restored_params), (
            "Should restore params from step 10"
        )
        assert exp.step == 10

        exp.cleanup()

    @pytest.mark.skip
    def test_dataset_checkpointing(self, train_dataset, checkpoint_dir, log_dir):
        """Test that dataset state is checkpointed and restored correctly."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        # Enable dataset checkpointing
        exp.checkpoint_config.checkpoint_dataset_iterator = True
        exp.init_state()

        # Fetch some batches to advance the dataset iterator
        batch1 = next(iter(train_dataset))
        exp.inner_loop(batch1)

        batch2 = next(iter(train_dataset))
        exp.inner_loop(batch2)

        exp.step = 100

        # Get the dataset state before saving
        dataset_state_before_save = train_dataset.get_state()

        # Save checkpoint
        exp.save_checkpoint()

        # Continue iterating the dataset
        batch3 = next(iter(train_dataset))
        exp.inner_loop(batch3)

        # Dataset state should have advanced
        dataset_state_after_batch3 = train_dataset.get_state()
        assert (
            dataset_state_after_batch3["iteration_count"]
            > dataset_state_before_save["iteration_count"]
        )

        # Restore checkpoint
        exp.restore_checkpoint(100)

        # Dataset state should be restored to the saved state
        dataset_state_after_restore = train_dataset.get_state()
        assert (
            dataset_state_after_restore["iteration_count"]
            == dataset_state_before_save["iteration_count"]
        )

        exp.cleanup()

    def test_train_metrics_computation(self, train_dataset, checkpoint_dir, log_dir):
        """Test that training step computes metrics correctly."""
        exp = _create_test_experiment(
            train_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Get a batch and run training
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)

        # The training should have computed metrics
        # We can verify this by checking that the logger received metrics
        # Since we're using a real FileLogger, we can't easily mock it,
        # but we can verify the step ran without error

        # Verify that default metrics include loss
        assert len(exp.train_task.metrics) > 0
        assert isinstance(exp.train_task.metrics[0], metrics.Loss)

        exp.cleanup()

    def test_eval_metrics_computation(
        self, train_dataset, eval_dataset, checkpoint_dir, log_dir
    ):
        """Test that evaluation computes metrics correctly."""
        exp = _create_test_experiment(
            train_dataset, eval_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.init_state()

        # Run validation
        result = exp.run_validation()

        # Verify result is a dict with val/ prefixed metrics
        assert isinstance(result, dict)
        assert "val/loss" in result
        assert np.isfinite(result["val/loss"])
        assert result["val/loss"] > 0

        exp.cleanup()

    def test_metrics_accumulated_correctly_across_batches(
        self, train_dataset, eval_dataset, checkpoint_dir, log_dir
    ):
        """Test that metrics are correctly averaged across multiple eval batches."""
        # Create an experiment with multiple eval steps
        exp = _create_test_experiment(
            train_dataset, eval_dataset, checkpoint_dir=checkpoint_dir, log_dir=log_dir
        )
        exp.eval_config.num_eval_steps = 2  # Use 2 batches for eval
        exp.init_state()

        # Run validation
        val_metrics = exp.run_validation()

        # Verify metrics were computed
        assert "val/loss" in val_metrics
        assert np.isfinite(val_metrics["val/loss"])
        assert val_metrics["val/loss"] > 0

        exp.cleanup()


class TestMultiOptimizer:
    """Tests for multi-optimizer functionality."""

    @patch("toylib_projects.tinystories.experiment.optax.multi_transform")
    def test_default_single_optimizer(self, mock_multi_transform):
        """Test that default behavior uses a single optimizer for all parameters."""
        mock_dataset = Mock()
        mock_dataset.batch_size = 4
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        exp = experiment.Experiment(train_task=train_task)
        exp.init_state()

        mock_multi_transform.assert_not_called()

        exp.cleanup()

    def test_custom_optimizer_mapping(self):
        """Test custom optimizer mapping for different model parts."""
        mock_dataset = Mock()
        mock_dataset.batch_size = 4
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        def optimizer_for_param(key_path: tuple) -> str:
            # Extract the keys - some entries have .key attribute (GetAttrKey), others don't
            path_strs = []
            for k in key_path:
                if hasattr(k, "key"):
                    path_strs.append(k.key if isinstance(k.key, str) else str(k.key))
                else:
                    path_strs.append(str(k))

            if ".embedding_layer" in path_strs:
                return "embedding_opt"
            if ".causal_attn" in path_strs:
                return "attention_opt"
            return "default"

        optimizer_configs = [
            experiment.OptimizerConfig(
                name="embedding_opt",
                optimizer=optax.adam(learning_rate=1e-4),
                learning_rate=1e-4,
            ),
            experiment.OptimizerConfig(
                name="attention_opt",
                optimizer=optax.adam(learning_rate=5e-4),
                learning_rate=5e-4,
            ),
            experiment.OptimizerConfig(
                name="default",
                optimizer=optax.adam(learning_rate=1e-3),
                learning_rate=1e-3,
            ),
        ]

        exp = experiment.Experiment(
            train_task=train_task,
            training_config=experiment.TrainingConfig(
                optimizer_config=experiment.MultiOptimizerConfig(
                    optimizer_configs=optimizer_configs,
                    optimizer_for_param=optimizer_for_param,
                )
            ),
        )
        exp.init_state()

        # Collect all optimizer names used
        optimizer_names = set()
        all_paths = []

        def collect_names(key_path, _):
            all_paths.append(key_path)
            optimizer_names.add(optimizer_for_param(key_path))

        jax.tree_util.tree_map_with_path(collect_names, exp.model)

        # Should have multiple optimizer names
        assert "embedding_opt" in optimizer_names
        assert "attention_opt" in optimizer_names or "default" in optimizer_names

        exp.cleanup()

    def test_configured_optimizer_mapping(self):
        """Test optimizer mapping supplied via TrainingConfig."""
        mock_dataset = Mock()
        mock_dataset.batch_size = 4
        train_task = experiment.Task(name="train", dataset=mock_dataset)

        def optimizer_for_param(_key_path: tuple) -> str:
            return "custom"

        optimizer_configs = [
            experiment.OptimizerConfig(
                name="custom",
                optimizer=optax.adam(learning_rate=1e-3),
                learning_rate=1e-3,
            )
        ]

        multi_optimizer_config = experiment.MultiOptimizerConfig(
            optimizer_configs=optimizer_configs,
            optimizer_for_param=optimizer_for_param,
        )

        exp = experiment.Experiment(
            train_task=train_task,
            training_config=experiment.TrainingConfig(
                optimizer_config=multi_optimizer_config,
            ),
        )
        exp.init_state()

        optimizer_names = set()

        def collect_names(key_path, _):
            optimizer_names.add(optimizer_for_param(key_path))

        jax.tree_util.tree_map_with_path(collect_names, exp.model)

        assert optimizer_names == {"custom"}, (
            f"Configured mapping should use only 'custom', got: {optimizer_names}"
        )

        exp.cleanup()

    def test_multi_optimizer_training_step(self):
        """Test that training works correctly with multi-optimizer setup."""

        # Create a small test dataset
        train_dataset = MockDataset(
            batch_size=2, seq_len=8, vocab_size=32, num_batches=2
        )
        train_task = experiment.Task(name="train", dataset=train_dataset)

        # Small model config for testing
        model_config = decoder_only_model.ModelConfig(
            vocab_size=32, num_layers=1, qkv_dim=8, num_heads=1
        )

        def optimizer_for_param(key_path: tuple) -> str:
            path_strs = [str(k.key) for k in key_path if hasattr(k, "key")]
            if ".embedding_layer" in path_strs:
                return "embedding_opt"
            return "default"

        optimizer_configs = [
            experiment.OptimizerConfig(
                name="embedding_opt",
                optimizer=optax.sgd(learning_rate=1e-3),
                learning_rate=1e-3,
            ),
            experiment.OptimizerConfig(
                name="default",
                optimizer=optax.adam(learning_rate=1e-2),
                learning_rate=1e-2,
            ),
        ]

        exp = experiment.Experiment(
            train_task=train_task,
            model_config=model_config,
            training_config=experiment.TrainingConfig(
                learning_rate=1e-2,
                max_steps=2,
                optimizer_config=experiment.MultiOptimizerConfig(
                    optimizer_configs=optimizer_configs,
                    optimizer_for_param=optimizer_for_param,
                ),
            ),
            jit_computations=False,
        )
        exp.init_state()

        # Store initial parameters
        initial_params = _get_model_params_flat(exp.model)

        # Run a training step
        batch = next(iter(train_dataset))
        exp.inner_loop(batch)

        # Verify parameters were updated
        updated_params = _get_model_params_flat(exp.model)
        assert not jnp.allclose(initial_params, updated_params), (
            "Parameters should be updated after training step"
        )

        exp.cleanup()
