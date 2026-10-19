"""In-memory datasets for the training harness.

Small datasets (MNIST, CIFAR-10, TinyShakespeare) fit in RAM as NumPy arrays.
``ArrayDataset`` serves them to ``Experiment`` as fixed-size, shuffled batches
and implements the harness's ``Dataset`` protocol, including checkpointable
iterator state.

Typical use::

    train = ArrayDataset({"image": x_train, "label": y_train}, batch_size=128)
    val = ArrayDataset({"image": x_test, "label": y_test}, batch_size=500,
                       shuffle=False, restart_each_iter=True)
"""

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class ArrayDataset:
    """Serve a dict of equal-length arrays as fixed-size batches.

    Each batch is a dict with the same keys as ``arrays``, every value sliced to
    ``batch_size`` rows along axis 0. The final partial batch of each epoch is
    dropped, so every step has the same shape and ``jit`` compiles once.

    One pass of ``for batch in dataset`` yields the rest of the current epoch.

    * ``restart_each_iter=False`` (training): iteration picks up where the
      previous one stopped, so an early ``break`` does not replay data and a
      restored checkpoint resumes mid-epoch. Each epoch reshuffles with a seed
      derived from ``(seed, epoch)``, so runs are reproducible.
    * ``restart_each_iter=True`` (evaluation): every ``iter()`` starts again
      from the first batch, so each evaluation sees the same examples.
    """

    arrays: dict[str, np.ndarray]
    batch_size: int
    shuffle: bool = True
    seed: int = 0
    restart_each_iter: bool = False

    def __post_init__(self) -> None:
        if not self.arrays:
            raise ValueError("ArrayDataset needs at least one array")
        lengths = {k: len(v) for k, v in self.arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All arrays must have the same length, got {lengths}")
        self.num_examples = next(iter(lengths.values()))
        if self.batch_size > self.num_examples:
            raise ValueError(
                f"batch_size {self.batch_size} is larger than the dataset "
                f"({self.num_examples} examples)"
            )
        # Iterator state: which epoch we are in, and the index of the next batch.
        self.epoch = 0
        self.position = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.num_examples // self.batch_size

    def _order(self, epoch: int) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.num_examples)
        return np.random.default_rng((self.seed, epoch)).permutation(self.num_examples)

    def __iter__(self) -> typing.Iterator[dict[str, np.ndarray]]:
        if self.restart_each_iter:
            self.epoch, self.position = 0, 0
        return self._iterate_epoch()

    def _iterate_epoch(self) -> typing.Iterator[dict[str, np.ndarray]]:
        if self.position >= self.batches_per_epoch:
            # The previous pass consumed the whole epoch but was stopped before
            # the generator could roll over (e.g. training hit max_steps).
            self.epoch, self.position = self.epoch + 1, 0
        order = self._order(self.epoch)
        while self.position < self.batches_per_epoch:
            start = self.position * self.batch_size
            idx = order[start : start + self.batch_size]
            # Advance before yielding: if the consumer stops after this batch,
            # the saved state already points at the next one.
            self.position += 1
            yield {k: v[idx] for k, v in self.arrays.items()}
        self.epoch, self.position = self.epoch + 1, 0

    def get_state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
