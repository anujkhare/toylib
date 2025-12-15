import abc
import dataclasses
import numpy as np
import jax
import typing


def _is_array(x: typing.Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


class Module(abc.ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.
    """

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make all Modules dataclasses
        cls = dataclasses.dataclass(cls, kw_only=True)
        # Automatically register subclasses as pytree nodes
        cls = jax.tree_util.register_pytree_node_class(cls)

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize all the trainable parameters in the module."""
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.Any:
        """Run a forward pass of the module."""
        pass

    def _get_trainable_param_keys(self) -> list[str]:
        """Get the list of attribute names that are trainable parameters."""
        param_keys = []
        for k, v in self.__dict__.items():
            if (
                (_is_array(v) and not _is_random_key(k))
                or isinstance(v, Module)
                or (
                    _is_supported_container(v)
                    and all(isinstance(elem, Module) for elem in v)
                )
            ):
                param_keys.append(k)
        return param_keys

    def __post_init__(self) -> None:
        self.init()
        self._trainable_param_keys = self._get_trainable_param_keys()

    def tree_flatten(self) -> tuple:
        params = []
        aux_data = dict()

        # Look through each attribute in the object
        for k, v in self.__dict__.items():
            if k not in self._trainable_param_keys:
                aux_data[k] = v
        for k in self._trainable_param_keys:
            v = self.__dict__[k]
            params.append(v)

        return params, aux_data

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> "Module":
        # Create a new empty object
        obj = object.__new__(cls)
        param_keys = static["_trainable_param_keys"]

        # overwrite all of the children using the values in the given pytree
        for k, v in zip(param_keys, dynamic):
            obj.__setattr__(k, v)

        for k, v in static.items():
            obj.__setattr__(k, v)

        return obj

    def __repr__(self) -> str:
        _, aux = self.tree_flatten()
        return str(aux)
