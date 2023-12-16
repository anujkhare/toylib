from abc import ABC
import jax
import numpy as np
from typing import Any
from jax.tree_util import register_pytree_with_keys

def _is_array(x: Any) -> bool:
    return isinstance(
        x, (jax.Array, np.ndarray, np.generic)
    ) or hasattr(x, "__jax_array__")
# , float, complex, bool, int

def _is_random_key(x: str) -> bool:
    return x == 'key'

class Module(ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.
    """
    def tree_flatten(self):
        dynamic = [(k, v) for k, v in self.__dict__.items() if _is_array(v) and not _is_random_key(k)]
        aux_ = {k: v for k, v in self.__dict__.items() if not _is_array(v) or _is_random_key(k)}
        aux_data = {'aux': aux_, 'dynamic_keys': [c[0] for c in dynamic]}

        return (dynamic, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Create a new empty object
        obj = object.__new__(cls)

        # overwrite all of the children using the values in the given pytree
        for k, v in aux_data['aux'].items():
            obj.__setattr__(k, v)
        for k, v in zip(aux_data['dynamic_keys'], children):
            obj.__setattr__(k, v)

        return obj


def register_toylib_module(cls):
    register_pytree_with_keys(cls, flatten_with_keys=cls.tree_flatten, unflatten_func=cls.tree_unflatten)