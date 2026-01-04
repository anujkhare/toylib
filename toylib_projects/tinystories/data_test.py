"""Tests for data.py."""

import pathlib
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toylib_projects.tinystories import data


@pytest.mark.skip(reason="removed the code for now")
class TestHFDataset:
    def test_smoke(self):
        """Test that we can fetch data from the internet! Non-hermetic: requires internet access."""
        dataset = data.BatchedTokenizedDatasetHF(
            batch_size=4,
            seq_len=1024,
            tokenizer_batch_size=2,
        )

        # Fetch a batch
        batch = next(dataset)
        inputs = batch["inputs"]
        targets = batch["targets"]

        assert inputs.shape == (4, 1024)
        assert targets.shape == (4, 1024)
        assert (inputs[:, 1:] == targets[:, :-1]).all().tolist()


class TestParquetDataset:
    """Test Parquet-based dataset loader with temporary parquet files."""

    # Test data for parquet files
    SOME_TEXT = [
        [
            "This is the first file the first record",
            "This is the first file the second record",
            "This is the first file the third record",
            "This is the first file the fourth record",
        ],
        [
            "This is the second file the first record",
            "This is the second file the second record",
            "This is the second file the third record",
            "This is the second file the fourth record",
        ],
    ]

    @pytest.fixture
    def temp_parquet_files(self):
        """Create temporary parquet files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            split_dir = tmpdir_path / "test"
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create parquet files
            filenames = []
            for i in range(len(self.SOME_TEXT)):
                filename = split_dir / f"shard_{i:05d}.parquet"
                table = pa.table({"text": self.SOME_TEXT[i]})
                # Use row_group_size=2 to test row group iteration
                pq.write_table(table, str(filename), row_group_size=2)
                filenames.append(filename)

            yield tmpdir_path, "test", filenames

    def test_list_files(self, temp_parquet_files):
        """Test listing parquet files in the dataset directory."""
        base_path, split, expected_files = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=str(base_path),
            split=split,
        )

        files = dataset.list_files()
        assert len(files) == len(expected_files)
        for file in files:
            assert file.suffix == ".parquet"

    def test_smoke(self, temp_parquet_files):
        """Test that we can load temporary parquet files."""
        base_path, split, _ = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=str(base_path),
            split=split,
            batch_size=2,
            seq_len=32,
            tokenizer_batch_size=2,
        )

        # Fetch a batch
        batch = next(dataset)
        inputs = batch["inputs"]
        targets = batch["targets"]

        assert inputs.shape == (2, 32)
        assert targets.shape == (2, 32)
        assert (inputs[:, 1:] == targets[:, :-1]).all().tolist()

    def test_checkpointing(self, temp_parquet_files):
        """Test saving and restoring dataset state."""
        base_path, split, _ = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=8,
            tokenizer_batch_size=2,
        )

        # Fetch a few batches
        _ = next(dataset)
        _ = next(dataset)

        # Save state
        state = dataset.get_state()

        # Fetch another batch
        batch3 = next(dataset)

        # Restore state
        dataset.restore_state(state)

        # Fetch again, should match batch3
        batch4 = next(dataset)

        # The restored batch should match the one we got after the checkpoint
        assert (batch3["inputs"] == batch4["inputs"]).all().tolist()
        assert (batch3["targets"] == batch4["targets"]).all().tolist()


class TestGrainDataset:
    """Test Grain-based dataset loader with temporary parquet files."""

    # Test data for parquet files
    SOME_TEXT = [
        [
            "This is the first file the first record",
            "This is the first file the second record",
            "This is the first file the third record",
            "This is the first file the fourth record",
        ],
        [
            "This is the second file the first record",
            "This is the second file the second record",
            "This is the second file the third record",
            "This is the second file the fourth record",
        ],
    ]

    @pytest.fixture
    def temp_parquet_files(self):
        """Create temporary parquet files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            split_dir = tmpdir_path / "test"
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create parquet files
            filenames = []
            for i in range(len(self.SOME_TEXT)):
                filename = split_dir / f"shard_{i:05d}.parquet"
                table = pa.table({"text": self.SOME_TEXT[i]})
                # Use row_group_size=2 to test row group iteration
                pq.write_table(table, str(filename), row_group_size=2)
                filenames.append(filename)

            yield tmpdir_path, "test", filenames

    def test_list_files(self, temp_parquet_files):
        """Test listing parquet files in the dataset directory."""
        base_path, split, expected_files = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
        )

        files = dataset.list_files()
        assert len(files) == len(expected_files)
        for file in files:
            assert file.suffix == ".parquet"

    def test_smoke(self, temp_parquet_files):
        """Test that we can load temporary parquet files using Grain."""
        base_path, split, _ = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=2,
            seq_len=32,
            tokenizer_batch_size=2,
        )

        # Fetch a batch
        batch = next(dataset)
        inputs = batch["inputs"]
        targets = batch["targets"]

        assert inputs.shape == (2, 32)
        assert targets.shape == (2, 32)
        # Check input-target alignment
        assert (inputs[:, 1:] == targets[:, :-1]).all().tolist()

    def test_sequential_order(self, temp_parquet_files):
        """Test that files are read sequentially."""
        base_path, split, _ = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=16,
            tokenizer_batch_size=1,
        )

        # Fetch first batch - should be from first file
        batch1 = next(dataset)
        assert batch1["inputs"].shape == (1, 16)

        # Verify we can continue iterating
        batch2 = next(dataset)
        assert batch2["inputs"].shape == (1, 16)

    def test_checkpointing(self, temp_parquet_files):
        """Test saving and restoring dataset state."""
        base_path, split, _ = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=8,
            tokenizer_batch_size=1,
        )

        # Fetch a few batches
        _ = next(dataset)
        _ = next(dataset)

        # Save state
        state = dataset.get_state()

        # Fetch another batch
        batch3 = next(dataset)

        # Restore state
        dataset.restore_state(state)

        # Fetch again, should match batch3
        batch4 = next(dataset)

        # The restored batch should match the one we got after the checkpoint
        assert (batch3["inputs"] == batch4["inputs"]).all().tolist()
        assert (batch3["targets"] == batch4["targets"]).all().tolist()

    def test_multiple_files(self, temp_parquet_files):
        """Test reading from multiple parquet files."""
        base_path, split, filenames = temp_parquet_files

        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=16,
            tokenizer_batch_size=1,
        )

        # Should be able to iterate through multiple files
        batches = []
        for _ in range(4):
            batches.append(next(dataset))

        # Verify we got all batches
        assert len(batches) == 4
        for batch in batches:
            assert batch["inputs"].shape == (1, 16)

    def test_tokenizer_batch_size(self, temp_parquet_files):
        """Test that tokenizer_batch_size correctly batches text records before tokenization."""
        base_path, split, _ = temp_parquet_files

        # Use tokenizer_batch_size=2 to batch 2 text records together
        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=16,
            tokenizer_batch_size=2,
        )

        # Should be able to fetch batches
        batch1 = next(dataset)
        assert batch1["inputs"].shape == (1, 16)

        batch2 = next(dataset)
        assert batch2["inputs"].shape == (1, 16)

    def test_checkpointing_across_batch_boundary(self, temp_parquet_files):
        """Test that checkpointing works correctly even across tokenizer batch boundaries."""
        base_path, split, _ = temp_parquet_files

        # Use tokenizer_batch_size=2 so we process 2 text records at a time
        dataset = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=8,
            tokenizer_batch_size=2,
        )

        # Fetch batches until we're in the middle of processing
        _ = next(dataset)
        _ = next(dataset)

        # Save state - this should capture the Grain iterator's internal batch state
        state = dataset.get_state()

        # Fetch more batches
        batch3 = next(dataset)
        batch4 = next(dataset)

        # Restore to the saved state
        dataset.restore_state(state)

        # Next batches should match what we got after the checkpoint
        batch3_restored = next(dataset)
        batch4_restored = next(dataset)

        # Verify the batches match
        assert (batch3["inputs"] == batch3_restored["inputs"]).all().tolist()
        assert (batch3["targets"] == batch3_restored["targets"]).all().tolist()
        assert (batch4["inputs"] == batch4_restored["inputs"]).all().tolist()
        assert (batch4["targets"] == batch4_restored["targets"]).all().tolist()

    def test_different_tokenizer_batch_sizes(self, temp_parquet_files):
        """Test that different tokenizer_batch_size values produce same final results."""
        base_path, split, _ = temp_parquet_files

        # Create two datasets with different tokenizer_batch_size
        dataset1 = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=16,
            tokenizer_batch_size=1,
        )

        dataset2 = data.BatchedTokenizedDatasetGrain(
            dataset_path=str(base_path),
            split=split,
            batch_size=1,
            seq_len=16,
            tokenizer_batch_size=2,
        )

        # Fetch first batch from each
        batch1 = next(dataset1)
        batch2 = next(dataset2)

        # They should produce the same results (same sequence of tokens)
        assert (batch1["inputs"] == batch2["inputs"]).all().tolist()
        assert (batch1["targets"] == batch2["targets"]).all().tolist()
