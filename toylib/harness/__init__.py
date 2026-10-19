"""Train, checkpoint, evaluate and run models.

* ``experiment``: the training loop (``Experiment``) and its config classes.
* ``data``: ``ArrayDataset``, in-memory arrays served as harness batches.
* ``inference``: list and restore checkpoints, run a model over a dataset.
* ``metrics`` and ``logger``: what gets measured, and where it is written.
* ``analyze``: parameter counts and compiled-memory estimates.

Needs the ``harness`` extra (``uv sync --extra harness``). Depends on
``toylib.nn`` only through the models it is given, never on ``toylib.viz``.
"""
