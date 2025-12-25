"""Tests for data.py."""

import pytest

from toylib_projects.tinystories import data


@pytest.mark.expensive
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
    base_path = "toylib_projects/tinystories/data/parquet/"
    split = "test"

    def test_list_files(self):
        """Test listing parquet files in the dataset directory."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=self.base_path,
            split=self.split,
        )

        files = dataset.list_files()
        assert len(files) == 1
        for file in files:
            assert file.suffix == ".parquet"

    def test_smoke(self):
        """Test that we can load a local parquet file."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=self.base_path,
            split=self.split,
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

    @pytest.mark.skip(reason="Not fully working yet")
    def test_checkpointing(self):
        """Test saving and restoring dataset state."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path=self.base_path,
            split=self.split,
            batch_size=2,
            seq_len=16,
            tokenizer_batch_size=2,
        )

        # Fetch a few batches
        print("batch 1")
        batch1 = next(dataset)
        print("batch 2")
        _ = next(dataset)

        # Save state
        print("checkpoint")
        state = dataset.get_state()

        # Fetch another batch
        print("batch 3")
        batch3 = next(dataset)

        # Restore state
        print("restore")
        dataset.restore_state(state)

        # Fetch again, should match batch3
        print("batch4")
        batch4 = next(dataset)

        print(state["file_index"], state["row_group_index"], len(state["token_buffer"]))
        assert (batch1["inputs"] == batch4["inputs"]).all().tolist()
        assert (batch1["targets"] == batch4["targets"]).all().tolist()
