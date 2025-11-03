"""Tests for data.py."""

from toylib_projects.tinystories import data


class TestHFDataset:
    def test_smoke(self):
        """Test that we can fetch data from the internet! Non-hermetic: requires internet access."""
        dataset = data.BatchedTokenizedHFDataset(
            bos_token=10000,
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
