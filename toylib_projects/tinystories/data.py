import abc
import dataclasses
import datasets as hf_datasets
import jax.numpy as jnp
import pathlib
import pyarrow.parquet as pq
import typing

import grain.python as grain

from transformers import AutoTokenizer


@dataclasses.dataclass
class DatasetState:
    """Serializable state for dataset checkpointing."""

    pass


@dataclasses.dataclass
class BatchedTokenizedDataset(abc.ABC):
    dataset_path: str = "karpathy/fineweb-edu-100b-shuffle"
    split: str = "train"

    tokenizer_name: str = "gpt2"
    seq_len: int = 2048

    tokenizer_batch_size: int = 8
    batch_size: int = 128

    @abc.abstractmethod
    def _get_dataset_iterator(self) -> typing.Iterator:
        raise NotImplementedError

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.bos_token = self.tokenizer.bos_token_id
        self.token_buffer = []

        self.dataset_iter = self._get_dataset_iterator()

    def __iter__(self):
        return self

    def __next__(self) -> jnp.ndarray:
        token_needed = self.batch_size * self.seq_len + 1  # 1 for the last target token
        while len(self.token_buffer) < token_needed:
            # Load tokenizer_batch_size sequences from the dataset
            input_batch = next(self.dataset_iter)
            texts = input_batch["text"]

            # Tokenize all sequences
            tokenized = self.tokenizer(
                texts,
                return_tensors=None,  # return a list of lists
                padding=False,
                truncation=False,
                max_length=None,
            )["input_ids"]

            # Add tokens to the buffer
            for tokens in tokenized:
                self.token_buffer.append(self.bos_token)
                self.token_buffer.extend(tokens)

        # Extract needed tokens from the buffer
        tokens = self.token_buffer[:token_needed]
        self.token_buffer = self.token_buffer[token_needed:]

        # Create jax arrays for inputs and targets
        inputs = jnp.array(tokens[:-1], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        targets = jnp.array(tokens[1:], dtype=jnp.uint16).reshape(
            self.batch_size, self.seq_len
        )
        return {
            "inputs": inputs,
            "targets": targets,
            "mask": jnp.ones_like(inputs),
        }

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing. Override in subclasses."""
        raise NotImplementedError("Checkpointing not supported for this dataset type")

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a checkpoint state. Override in subclasses."""
        raise NotImplementedError("Checkpointing not supported for this dataset type")


class BatchedTokenizedDatasetHF(BatchedTokenizedDataset):
    def _get_dataset_iterator(self):
        # Load the dataset and fetch in batches
        return iter(
            hf_datasets.load_dataset(
                self.dataset_path, streaming=True, split=self.split
            ).batch(self.tokenizer_batch_size)
        )


@dataclasses.dataclass
class DatasetStateParquet(DatasetState):
    file_index: int = 0
    row_group_index: int = 0
    token_buffer: list[int] = dataclasses.field(default_factory=list)


class BatchedTokenizedDatasetParquet(BatchedTokenizedDataset):
    """Path is constructed as dataset_path/split/*.parquet"""

    def __post_init__(self):
        # Initialize position tracking before calling parent
        self._state = DatasetStateParquet()
        super().__post_init__()

    def list_files(self) -> list[pathlib.Path]:
        base_path = pathlib.Path(self.dataset_path) / self.split
        # Sort for deterministic ordering across runs
        return sorted(base_path.glob("*.parquet"))

    def _get_dataset_iterator(self) -> typing.Iterator:
        """Generator that tracks position for checkpointing."""
        files = self.list_files()

        # Start from the tracked position
        for file_idx in range(self._state.file_index, len(files)):
            self._state.file_index = file_idx
            pf = pq.ParquetFile(files[file_idx])

            # Read row groups from the current file
            for rg_idx in range(self._state.row_group_index, pf.num_row_groups):
                self._state.row_group_index = rg_idx
                rg = pf.read_row_group(rg_idx)
                yield {"text": rg.column("text").to_pylist()}

            # Reset row group index for next file
            self._state.row_group_index = 0

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing."""
        self._state.token_buffer = self.token_buffer.copy()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore iterator position from checkpoint."""
        self._state = DatasetStateParquet(**state)
        self._state.token_buffer = state["token_buffer"].copy()

        # Recreate the iterator starting from the restored position
        self.dataset_iter = self._get_dataset_iterator()


@dataclasses.dataclass
class DatasetStateGrain(DatasetState):
    """State for Grain-based dataset checkpointing."""

    sampler_state: dict = dataclasses.field(default_factory=dict)
    token_buffer: list[int] = dataclasses.field(default_factory=list)


class BatchedTokenizedDatasetGrain(BatchedTokenizedDataset):
    """Grain-based data loader that reads parquet files.

    Uses Grain's efficient data loading pipeline for reading parquet files.
    Path is constructed as dataset_path/split/*.parquet

    Example:
        >>> dataset = BatchedTokenizedDatasetGrain(
        ...     dataset_path="/path/to/data",
        ...     split="train",
        ...     batch_size=128,
        ...     seq_len=2048,
        ... )
        >>> for batch in dataset:
        ...     train_step(batch)
    """

    seed: int = 42

    def __post_init__(self):
        # Initialize state tracking before calling parent
        self._state = DatasetStateGrain()
        super().__post_init__()

    def list_files(self) -> list[pathlib.Path]:
        """List parquet files for the split."""
        base_path = pathlib.Path(self.dataset_path) / self.split
        return sorted(base_path.glob("*.parquet"))

    def _get_dataset_iterator(self) -> typing.Iterator:
        """Create Grain-based iterator over parquet files."""
        # Create a MapDataset from the file paths
        dataset = grain.MapDataset.source([str(f) for f in self.list_files()])

        # Map each filename to a ParquetIterDataset
        dataset = dataset.map(grain.experimental.ParquetIterDataset)

        # Interleave with cycle_length=1 to chain datasets sequentially
        dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=1)

        # Batch records at the row level (groups individual text records together)
        dataset = dataset.batch(self.tokenizer_batch_size, drop_remainder=False)

        # Create iterator
        iterator = iter(dataset)

        # Restore state if available
        if self._state.sampler_state:
            iterator.set_state(self._state.sampler_state)

        # Store iterator for state management
        self._grain_iterator = iterator

        # Yield batches in the format expected by parent class
        for batch in iterator:
            # batch is a dict with the key "text" containing an array of texts
            yield {"text": list(batch["text"])}

    def get_state(self) -> dict[str, typing.Any]:
        """Get current state for checkpointing.

        Returns:
            Dictionary containing iterator state and token buffer.
        """
        # Save the current iterator state
        self._state.sampler_state = self._grain_iterator.get_state()
        self._state.token_buffer = self.token_buffer.copy()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a checkpoint state.

        Args:
            state: State dictionary containing sampler state and token buffer.
        """
        self._state = DatasetStateGrain(**state)
        self._state.token_buffer = state["token_buffer"].copy()

        # Recreate the iterator with the restored state
        self.dataset_iter = self._get_dataset_iterator()
