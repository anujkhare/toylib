"""Tests for data.py."""

from toylib_projects.tinystories import data


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
    def test_list_files(self):
        """Test listing parquet files in the dataset directory."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path="toylib_projects/tinystories/data/parquet/",
            split="test",
        )

        files = dataset.list_files()
        assert len(files) == 1
        for file in files:
            assert file.suffix == ".parquet"

    def test_smoke(self):
        """Test that we can fetch data from the internet! Non-hermetic: requires internet access."""
        dataset = data.BatchedTokenizedDatasetParquet(
            dataset_path="toylib_projects/tinystories/data/parquet/",
            split="test",
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
