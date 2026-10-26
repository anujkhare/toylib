"""Shared data plumbing: download a file once into ``~/.cache/toylib`` and reuse it.

Dataset-specific loaders (MNIST, CIFAR-10, ...) live with the project that uses
them, e.g. ``toylib_projects/p01_cnn_mnist/mnist.py``.
"""
