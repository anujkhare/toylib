import abc
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import typing


def _is_array(x: typing.Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


def _wrap_init(orig: typing.Callable) -> typing.Callable:
    def wrapped(self) -> None:
        orig(self)
        # DFS: init any sub-modules created during orig(self)
        for v in self.__dict__.values():
            if isinstance(v, Module) and not hasattr(v, "_trainable_param_keys"):
                v.init()
            elif _is_supported_container(v):
                for elem in v:
                    if isinstance(elem, Module) and not hasattr(
                        elem, "_trainable_param_keys"
                    ):
                        elem.init()
        self._trainable_param_keys = self._get_trainable_param_keys()
        # Drop the random key after init since it's not needed anymore.
        if hasattr(self, "key"):
            self.key = None

    return wrapped


@dataclasses.dataclass
class Module(abc.ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.

    Every subclass automatically receives two dtype fields inherited from this base:

        param_dtype: storage dtype for trainable parameters (default float32).
        dtype: compute dtype for forward-pass operations (default float32).
    """

    param_dtype: np.dtype | type = jnp.float32
    dtype: np.dtype | type = jnp.float32

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        """Initialize subclass as a dataclass and register as a pytree node.

        Sub-classes of dataclasses are not automatically dataclasses, so we need to explicitly convert them.
        We also register the class as a pytree with jax so that it can be used with jax transformations like jit and grad.

        Also wraps the subclass's init() to recursively initialize any sub-Module instances
        created during init(), then compute _trainable_param_keys. This means calling init()
        on the top-level module is sufficient to initialize the entire module tree.
        """
        super().__init_subclass__(**kwargs)
        # Make all Modules dataclasses.
        cls = dataclasses.dataclass(cls, kw_only=True)
        # Automatically register subclasses as pytree nodes
        cls = jax.tree_util.register_pytree_with_keys_class(cls)
        # Wrap init() to recursively init sub-modules and compute _trainable_param_keys
        if "init" in cls.__dict__:
            original_init = cls.__dict__["init"]

            cls.init = _wrap_init(original_init)

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

    def tree_flatten_with_keys(self) -> tuple:
        params_with_keys = []
        aux_data = dict()

        # Look through each attribute in the object
        for k, v in self.__dict__.items():
            if k not in self._trainable_param_keys:
                aux_data[k] = v
        for k in self._trainable_param_keys:
            v = self.__dict__[k]
            params_with_keys.append((jax.tree_util.GetAttrKey(k), v))

        return params_with_keys, aux_data

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
