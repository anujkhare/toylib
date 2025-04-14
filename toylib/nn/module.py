from typing import Any
import numpy as np
import jax
import typing


def _is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


@jax.tree_util.register_pytree_node_class
class Module:
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.
    """

    def tree_flatten(self) -> tuple:
        params = []
        param_keys = []
        aux_data = dict()

        # Look through each attribute in the object
        for k, v in self.__dict__.items():
            if (
                (_is_array(v) and not _is_random_key(k))
                or isinstance(v, Module)
                or (
                    _is_supported_container(v)
                    and all(isinstance(elem, Module) for elem in v)
                )
            ):
                # trainable leaf param!
                params.append(v)
                param_keys.append(k)
            else:
                aux_data[k] = v

        aux_data["param_keys"] = param_keys
        return params, aux_data

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> "Module":
        # Create a new empty object
        obj = object.__new__(cls)

        # overwrite all of the children using the values in the given pytree
        for k, v in zip(static["param_keys"], dynamic):
            obj.__setattr__(k, v)

        for k, v in static.items():
            obj.__setattr__(k, v)

        return obj

    def __repr__(self) -> str:
        _, aux = self.tree_flatten()
        return str(aux)
