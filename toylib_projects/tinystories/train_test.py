"""Tests for train.py

These tests ensure basic coverage for API changes in the modules used by train.py.
Since the underlying modules (experiment, decoder_only_model, data, etc.) are unit
tested separately, these tests focus on integration and API compatibility.
"""

from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toylib_projects.tinystories import experiment
from toylib_projects.tinystories import train

VOCAB_SIZE = 50257


def create_dummy_dataset(
    dataset_path: Path, split: str, num_files: int = 2, rows_per_file: int = VOCAB_SIZE
):
    """Create dummy parquet files that mimic a text dataset.

    Args:
        dataset_path: Base path for the dataset
        split: Split name (e.g., 'train', 'val')
        num_files: Number of parquet files to create
        rows_per_file: Number of text rows per file
    """
    split_dir = dataset_path / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for file_idx in range(num_files):
        # Generate dummy text data
        texts = [
            f"This is sample text number {i} in file {file_idx}. "
            f"It contains some words to tokenize. " * 5
            for i in range(rows_per_file)
        ]

        # Create a PyArrow table with a 'text' column
        table = pa.table({"text": texts})

        # Write to parquet
        output_path = split_dir / f"data_{file_idx:04d}.parquet"
        pq.write_table(table, output_path)


def create_dummy_bpt_mapping(bpt_path: Path, vocab_size: int = VOCAB_SIZE):
    """Create a dummy bytes-per-token mapping file.

    Args:
        bpt_path: Path where to save the .npy file
        vocab_size: Vocabulary size (default is GPT-2 vocab size)
    """
    # Create a random mapping (in reality this would be based on actual tokenizer)
    # Most tokens are 1-4 bytes in UTF-8
    bpt_mapping = np.random.randint(1, 5, size=vocab_size)
    np.save(bpt_path, bpt_mapping)


class TestCreateExperiment:
    """Tests for create_experiment function."""

    @pytest.fixture
    def fake_fs(self, tmp_path: Path):
        """Create a fake filesystem with dummy dataset and bpt mapping."""
        dataset_path = tmp_path / "dataset"
        checkpoint_dir = tmp_path / "checkpoints"
        bpt_path = tmp_path / "bpt_gpt2.npy"

        # Create dummy dataset files
        create_dummy_dataset(dataset_path, split="train", num_files=2, rows_per_file=5)
        create_dummy_dataset(dataset_path, split="val", num_files=1, rows_per_file=3)

        # Create dummy bytes-per-token mapping
        create_dummy_bpt_mapping(bpt_path, vocab_size=VOCAB_SIZE)

        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return {
            "dataset_path": str(dataset_path),
            "checkpoint_dir": str(checkpoint_dir),
            "bpt_path": str(bpt_path),
        }

    def test_create_experiment_with_fake_filesystem(self, fake_fs):
        """Test that create_experiment works with dummy dataset and files."""
        # Create a tiny model to speed up the test
        exp = train.create_experiment(
            depth=1,  # Very small model
            vocab_size=VOCAB_SIZE,  # Use default GPT-2 vocab
            batch_size_per_device=2,
            seq_len=64,  # Short sequences
            max_steps=1,
            num_microbatches=1,
            checkpoint_dir=fake_fs["checkpoint_dir"],
            dataset_path=fake_fs["dataset_path"],
            dataset_train_split="train",
            dataset_val_split="val",
            bpt_path=fake_fs["bpt_path"],
        )

        # Verify experiment was created
        assert exp is not None
        assert isinstance(exp, experiment.Experiment)
        assert exp.model is not None
        assert exp.optimizer is not None
        assert exp.opt_state is not None

        # Verify tasks were created
        assert exp.train_task is not None
        assert exp.eval_task is not None
        assert exp.train_task.name == "train"
        assert exp.eval_task.name == "val"

        # Verify configs
        assert exp.training_config.max_steps == 1
        assert exp.model_config.vocab_size == VOCAB_SIZE

        exp.cleanup()

    def test_create_experiment_without_validation_split(self, fake_fs):
        """Test create_experiment when validation split is None."""
        exp = train.create_experiment(
            depth=1,
            vocab_size=VOCAB_SIZE,
            batch_size_per_device=2,
            seq_len=64,
            max_steps=1,
            num_microbatches=1,
            checkpoint_dir=fake_fs["checkpoint_dir"],
            dataset_path=fake_fs["dataset_path"],
            dataset_train_split="train",
            dataset_val_split=None,  # No validation
            bpt_path=fake_fs["bpt_path"],
        )

        assert exp is not None
        assert exp.train_task is not None
        assert exp.eval_task is None  # Should be None

        exp.cleanup()

    def test_experiment_can_run_one_step(self, fake_fs):
        """Test that the experiment can actually run a training step."""
        exp = train.create_experiment(
            depth=1,
            vocab_size=VOCAB_SIZE,
            batch_size_per_device=2,
            seq_len=64,
            max_steps=1,
            num_microbatches=1,
            checkpoint_dir=fake_fs["checkpoint_dir"],
            dataset_path=fake_fs["dataset_path"],
            dataset_train_split="train",
            dataset_val_split=None,
            bpt_path=fake_fs["bpt_path"],
        )

        # Run the outer loop (which should do 1 step)
        exp.outer_loop()

        # Verify training happened
        assert exp.step >= 1

        exp.cleanup()

    def test_multi_optimizer_config(self):
        """Test that multi-optimizer configuration is created correctly."""
        config = train.create_muon_adam_multi_optimizer_config(
            muon_lr=1e-4,
            adamw_embed_lr=2e-4,
            adamw_output_lr=3e-4,
            weight_decay=0.01,
        )

        assert config is not None
        assert isinstance(config, experiment.MultiOptimizerConfig)
        assert len(config.optimizer_configs) == 3
        assert config.optimizer_for_param is not None

        # Check optimizer names
        names = {opt.name for opt in config.optimizer_configs}
        assert "muon" in names
        assert "adamw_embed" in names
        assert "adamw_output" in names
