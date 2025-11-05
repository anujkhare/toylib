import abc
import dataclasses
import datasets as hf_datasets
import jax.numpy as jnp
import pathlib
import pyarrow.parquet as pq
import typing

from transformers import AutoTokenizer


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
        }


class BatchedTokenizedDatasetHF(BatchedTokenizedDataset):
    def _get_dataset_iterator(self):
        # Load the dataset and fetch in batches
        return iter(
            hf_datasets.load_dataset(
                self.dataset_path, streaming=True, split=self.split
            ).batch(self.tokenizer_batch_size)
        )


class BatchedTokenizedDatasetParquet(BatchedTokenizedDataset):
    """Path is constructed as dataset_path/split/*.parquet"""

    def list_files(self):
        base_path = pathlib.Path(self.dataset_path) / self.split
        return list(base_path.glob("*.parquet"))

    # TODO(anujkhare): does not respect tokenizer_batch_size yet
    def _get_dataset_iterator(self):
        for file_path in self.list_files():
            pf = pq.ParquetFile(file_path)
            for row_group in range(pf.num_row_groups):
                rg = pf.read_row_group(row_group)
                yield {"text": rg.column("text").to_pylist()}
