"""Tests for module.py."""

import jax.numpy as jnp

from toylib.nn import module


class TestModule:
    def test_smoke(self):
        class Foo(module.Module):
            a: int = 1
            b: str = "hello"

            def init(self) -> None:
                pass

            def __call__(self) -> None:
                pass

        obj = Foo()
        obj.a = jnp.array([1, 2, 3])
        obj.b = "hello"
        params, aux = obj.tree_flatten()
        unflattened = Foo.tree_unflatten(aux, params)
        assert jnp.array_equal(unflattened.a, obj.a)
        assert unflattened.b == obj.b
