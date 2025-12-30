"""Tests for module.py."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from toylib.nn import module


class TestModule:
    @pytest.fixture
    def linear_module(self):
        """Fixture that provides a test module."""

        class Linear(module.Module):
            in_features: int = 3
            out_features: int = 2
            key: jt.PRNGKeyArray

            def init(self) -> None:
                self.w = jax.random.uniform(
                    key=self.key, shape=(self.in_features, self.out_features)
                )
                self.bias = jnp.zeros((self.out_features,))

            def __call__(self, x):
                return x @ self.w + self.bias

        return Linear

    @pytest.fixture
    def sample_input(self):
        """Fixture that provides sample input data."""
        return jnp.array([1.0, 2.0, 3.0])

    def test_tree_flatten_unflatten_roundtrip(self):
        """Test that flatten/unflatten preserves the module."""

        class Foo(module.Module):
            b: str = "hello"

            def init(self) -> None:
                self.a = jnp.array([1, 2, 3], dtype=jnp.float32)

            def __call__(self, x):
                return self.a + x

        obj = Foo(b="world")

        # Use jax.tree_util functions which internally use your methods
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        unflattened = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.array_equal(unflattened.a, obj.a)
        assert unflattened.b == obj.b

    def test_jit(self, linear_module, sample_input):
        """Test that module works with jax.jit."""

        @jax.jit
        def forward(model, x):
            return model(x)

        model = linear_module(key=jax.random.PRNGKey(0))
        result = forward(model, sample_input)
        expected = sample_input @ model.w + model.bias
        assert jnp.array_equal(result, expected)

    def test_grad(self, linear_module, sample_input):
        """Test that module works with jax.grad."""

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        model = linear_module(key=jax.random.PRNGKey(0))
        grads = jax.grad(loss_fn)(model, sample_input)

        # grads should be a Linear module with gradient arrays
        assert isinstance(grads, linear_module)
        assert grads.w.shape == model.w.shape
        assert grads.bias.shape == model.bias.shape

    def test_value_and_grad(self, linear_module, sample_input):
        """Test that module works with jax.value_and_grad."""

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        model = linear_module(key=jax.random.PRNGKey(0))
        loss, grads = jax.value_and_grad(loss_fn)(model, sample_input)

        assert isinstance(loss, jax.Array)
        assert isinstance(grads, linear_module)

    def test_tree_map_with_path(self):
        """Test that keys are preserved for path-aware operations."""

        class Foo(module.Module):
            def init(self) -> None:
                self.weight = jnp.array([1.0, 2.0])

            def __call__(self, x):
                return x * self.weight

        obj = Foo()
        paths = []

        def collect_paths(path, leaf):
            paths.append(path)
            return leaf

        jax.tree_util.tree_map_with_path(collect_paths, obj)

        # Should have captured the path to 'weight'
        assert len(paths) > 0
