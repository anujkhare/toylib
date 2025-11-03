import datasets as hf_datasets
import jax.numpy as jnp
from transformers import AutoTokenizer


class BatchedTokenizedHFDataset:
    def __init__(
        self,
        dataset_path: str = "karpathy/fineweb-edu-100b-shuffle",
        bos_token: int = -1,
        tokenizer_name: str = "gpt2",
        split: str = "train",
        *,
        batch_size: int = 128,
        seq_len: int = 2048,
        tokenizer_batch_size: int = 8,
        streaming: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_iter = iter(
            hf_datasets.load_dataset(
                dataset_path, streaming=streaming, split=split
            ).batch(tokenizer_batch_size)
        )  # Fetch in batches

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bos_token = bos_token

        self.token_buffer = []

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
