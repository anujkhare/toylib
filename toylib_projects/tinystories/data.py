import datasets as hf_datasets
import functools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class TextDataset(Dataset):
    def __init__(self, input_npy_file: str, *, max_tokens=1024):
        self.max_tokens = max_tokens
        self._input_npy_file = input_npy_file
        self.data = np.load(input_npy_file, allow_pickle=True)

    def _pad_or_truncate(self, tokens: list[int]) -> tuple[np.array, np.ndarray]:
        if len(tokens) < self.max_tokens:
            mask = np.array(
                [1] * len(tokens) + [0] * (self.max_tokens - len(tokens)), dtype=np.bool
            )
            # Pad with zeros
            tokens = tokens + [0] * (self.max_tokens - len(tokens))
        else:
            mask = np.ones(self.max_tokens, dtype=np.bool)
            # Truncate
            tokens = tokens[: self.max_tokens]
        return np.array(tokens), mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Pad or truncate
        tokens, mask = self._pad_or_truncate(item["tokens"])
        return {
            "tokens": tokens,
            "mask": mask,
            "num_tokens": len(tokens),
        }


def numpy_collate(batch):
    return {
        "tokens": np.stack([item["tokens"] for item in batch]),  # (B, T),
        "mask": np.stack([item["mask"] for item in batch]),  # (B, T),
        "num_tokens": np.array([item["num_tokens"] for item in batch]),  # (B,)
    }


def load_dataset(
    batch_size, max_tokens, train_npy_file: str = None, valid_npy_file: str = None
):
    # Create Dataset classes
    dataset_fn = functools.partial(
        TextDataset,
        max_tokens=max_tokens,
    )
    train_dataset, train_dataloader = None, None
    val_dataset, val_dataloader = None, None

    if train_npy_file is not None:
        train_dataset = dataset_fn(train_npy_file)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )

    if valid_npy_file is not None:
        val_dataset = dataset_fn(valid_npy_file)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )

    return train_dataset, train_dataloader, val_dataset, val_dataloader
