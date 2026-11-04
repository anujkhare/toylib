"""Tests for module.py."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
import pytest

from toylib.nn import module


class TestModule:
    @pytest.fixture
    def linear_module(self):
        """Fixture that provides a test module."""

        class Linear(module.Module):
            key: jt.PRNGKeyArray
            in_features: int = 3
            out_features: int = 2

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
        obj.init()

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

        model = linear_module(key=jax.random.key(0))
        model.init()
        result = forward(model, sample_input)
        expected = sample_input @ model.w + model.bias
        assert jnp.array_equal(result, expected)

    def test_grad(self, linear_module, sample_input):
        """Test that module works with jax.grad."""

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        model = linear_module(key=jax.random.key(0))
        model.init()
        grads = jax.grad(loss_fn)(model, sample_input)

        # grads should be a Linear module with gradient arrays
        assert isinstance(grads, linear_module)
        assert grads.w.shape == model.w.shape
        assert grads.bias.shape == model.bias.shape

    def test_value_and_grad(self, linear_module, sample_input):
        """Test that module works with jax.value_and_grad."""

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        model = linear_module(key=jax.random.key(0))
        model.init()
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
        obj.init()
        paths = []

        def collect_paths(path, leaf):
            paths.append(path)
            return leaf

        jax.tree_util.tree_map_with_path(collect_paths, obj)

        # Should have captured the path to 'weight'
        assert len(paths) > 0

    def test_optax_training_loop(self, linear_module, sample_input):
        """Test a full training loop with optax."""
        model = linear_module(key=jax.random.key(0))
        model.init()
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(model)

        target = jnp.array([1.0, 0.0])

        def loss_fn(model, inputs):
            return jnp.mean((model(inputs) - target) ** 2)

        @jax.jit
        def train_step(m, opt_state, x):
            loss, grads = jax.value_and_grad(loss_fn)(m, x)
            updates, new_opt_state = optimizer.update(grads, opt_state, m)
            new_model = optax.apply_updates(m, updates)
            return new_model, new_opt_state, loss

        new_model, opt_state, loss = train_step(model, opt_state, sample_input)

        assert isinstance(new_model, linear_module)
        assert not jnp.array_equal(new_model.w, model.w)
        assert not jnp.array_equal(new_model.bias, model.bias)

        assert loss > 0


class _Inner(module.Module):
    def init(self) -> None:
        self.w = jnp.ones((2,))

    def __call__(self, x):
        return x * self.w


class _Outer(module.Module):
    def init(self) -> None:
        self.trained = jnp.zeros((3,))
        self.table = jnp.arange(4.0)
        self.encoder = _Inner()
        self.blocks = [_Inner(), _Inner()]
        self.freeze("table", "encoder")

    def __call__(self, x):
        return x


class TestFreeze:
    def _model(self):
        model = _Outer()
        model.init()
        return model

    def test_mask_marks_frozen_arrays_and_sub_modules(self):
        mask = module.trainable_mask(self._model())
        assert mask.trained is True
        assert mask.table is False
        assert mask.encoder.w is False
        assert [block.w for block in mask.blocks] == [True, True]

    def test_mask_has_the_model_structure(self):
        model = self._model()
        assert jax.tree.structure(module.trainable_mask(model)) == jax.tree.structure(
            jax.tree.map(lambda _: True, model)
        )

    def test_frozen_arrays_stay_leaves(self):
        """Frozen arrays go wherever the model goes: jit, checkpoints, sharding."""
        assert len(jax.tree.leaves(self._model())) == 5

    def test_frozen_keys_survive_flatten_and_unflatten(self):
        model = jax.tree.map(lambda x: x + 1, self._model())
        assert jax.tree.leaves(module.trainable_mask(model)) == [True, False, False, True, True]

    def test_mask_works_under_jit(self):
        @jax.jit
        def frozen_sum(model):
            _, frozen = module.partition(model, module.trainable_mask(model))
            return sum(x.sum() for x in jax.tree.leaves(frozen))

        assert frozen_sum(self._model()) == 6.0 + 2.0  # table and encoder.w

    def test_partition_then_combine_round_trips(self):
        model = self._model()
        trainable, frozen = module.partition(model, module.trainable_mask(model))
        assert len(jax.tree.leaves(trainable)) == 3
        assert len(jax.tree.leaves(frozen)) == 2
        restored = module.combine(trainable, frozen)
        assert jax.tree.structure(restored) == jax.tree.structure(model)
        for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(model)):
            assert jnp.array_equal(a, b)

    def test_trees_without_modules_are_trainable(self):
        assert module.trainable_mask({"w": jnp.zeros(2)}) == {"w": True}

    def test_rejects_a_name_that_is_not_a_parameter(self):
        class Bad(module.Module):
            size: int = 3

            def init(self) -> None:
                self.freeze("size")

            def __call__(self, x):
                return x

        with pytest.raises(ValueError, match="not a parameter"):
            Bad().init()
