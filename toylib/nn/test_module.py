"""Tests for module.py."""

import jax.numpy as jnp

from toylib.nn import module


class TestModule:
    def test_smoke(self):
        obj = module.Module()
        obj.a = jnp.array([1, 2, 3])
        obj.b = "hello"
        params, aux = obj.tree_flatten()
        unflattened = module.Module.tree_unflatten(aux, params)
        assert jnp.array_equal(unflattened.a, obj.a)
        assert unflattened.b == obj.b
