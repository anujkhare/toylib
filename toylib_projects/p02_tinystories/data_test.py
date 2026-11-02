"""Tests for data.py."""

import pathlib
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toylib_projects.p02_tinystories import data


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


class TestParquetEpochs:
    """The harness loops over a dataset once per epoch and once per eval."""

    @pytest.fixture
    def dataset_path(self, tmp_path):
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        rows = [f"story {i} has a few words in it" for i in range(8)]
        pq.write_table(pa.table({"text": rows}), split_dir / "a.parquet", row_group_size=2)
        return str(tmp_path)

    def test_second_epoch_has_batches(self, dataset_path):
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=dataset_path, batch_size=1, seq_len=8, tokenizer_batch_size=2
        )
        first = [np.asarray(b["inputs"]) for b in dataset]
        second = [np.asarray(b["inputs"]) for b in dataset]
        assert len(first) > 0
        assert len(first) == len(second)
        np.testing.assert_array_equal(first[0], second[0])

    def test_repeatable_restarts_on_iter(self, dataset_path):
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=dataset_path,
            batch_size=1,
            seq_len=8,
            tokenizer_batch_size=2,
            repeatable=True,
        )
        first = next(iter(dataset))
        second = next(iter(dataset))
        np.testing.assert_array_equal(first["inputs"], second["inputs"])

    def test_batches_are_contiguous(self, dataset_path):
        """No token falls between one batch's targets and the next batch's inputs."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=dataset_path, batch_size=1, seq_len=8, tokenizer_batch_size=2
        )
        first, second = next(dataset), next(dataset)
        assert first["targets"][0, -1] == second["inputs"][0, 0]


class TestTokenBinDataset:
    @pytest.fixture
    def path(self, tmp_path):
        path = tmp_path / "tokens.bin"
        np.arange(1000, dtype=np.uint16).tofile(path)
        return str(path)

    def test_targets_are_inputs_shifted_by_one(self, path):
        batch = next(iter(data.TokenBinDataset(path, batch_size=4, seq_len=16)))
        assert batch["inputs"].shape == (4, 16)
        np.testing.assert_array_equal(batch["targets"], batch["inputs"] + 1)
        assert batch["mask"].all()

    def test_rows_come_from_different_places(self, path):
        batch = next(iter(data.TokenBinDataset(path, batch_size=4, seq_len=16)))
        assert len(set(batch["inputs"][:, 0].tolist())) > 1

    def test_never_runs_past_the_end(self, path):
        dataset = iter(data.TokenBinDataset(path, batch_size=64, seq_len=999))
        for _ in range(20):
            batch = next(dataset)
        assert batch["targets"].max() == 999

    def test_batches_differ_until_restarted_if_repeatable(self, path):
        dataset = data.TokenBinDataset(path, batch_size=4, seq_len=16, repeatable=True)
        iterator = iter(dataset)
        first, second = next(iterator), next(iterator)
        assert not np.array_equal(first["inputs"], second["inputs"])
        np.testing.assert_array_equal(first["inputs"], next(iter(dataset))["inputs"])

    def test_training_stream_continues_across_iters(self, path):
        """Each harness epoch loop must see new batches, not the same ones again."""
        dataset = data.TokenBinDataset(path, batch_size=4, seq_len=16)
        first = next(iter(dataset))
        second = next(iter(dataset))
        assert not np.array_equal(first["inputs"], second["inputs"])

    def test_resume_gives_the_same_next_batch(self, path):
        dataset = data.TokenBinDataset(path, batch_size=4, seq_len=16)
        iterator = iter(dataset)
        next(iterator)
        state = dataset.get_state()
        expected = next(iterator)

        resumed = data.TokenBinDataset(path, batch_size=4, seq_len=16)
        resumed.restore_state(state)
        np.testing.assert_array_equal(expected["inputs"], next(iter(resumed))["inputs"])

    def test_rejects_a_file_shorter_than_a_window(self, path):
        with pytest.raises(ValueError, match="shorter"):
            data.TokenBinDataset(path, batch_size=1, seq_len=1000)
