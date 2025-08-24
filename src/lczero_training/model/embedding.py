import math

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import initializers as flax_initializers
from flax.typing import Initializer, Shape

from proto import net_pb2

from .utils import get_activation


class Embedding(nnx.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        dense_size: int,
        embedding_size: int,
        dff: int,
        default_activation: net_pb2.NetworkFormat.ActivationFunction,
        ffn_activation: net_pb2.NetworkFormat.ActivationFunction,
        rngs: nnx.Rngs,
    ):
        self._input_channels = input_channels
        self.dense_size = dense_size
        self.default_activation = default_activation

        assert dense_size > 0
        self.preprocess = nnx.Linear(
            in_features=64 * 12,
            out_features=64 * dense_size,
            rngs=rngs,
        )

        assert embedding_size > 0
        self.square_embedding = nnx.Linear(
            in_features=input_channels + dense_size,
            out_features=embedding_size,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embedding_size, rngs=rngs)
        self.ma_gating = MaGating(feature_shape=(embedding_size,), rngs=rngs)
        self.alpha = math.pow(2.0 * embedding_size, -0.25)
        beta = math.pow(8.0 * embedding_size, -0.25)
        self.ffn = Ffn(
            in_features=embedding_size,
            layer1_features=dff,
            layer2_features=embedding_size,
            layer1_activation=ffn_activation,
            kernel_init=scaled_xavier_normal(beta),
            rngs=rngs,
        )
        self.out_norm = nnx.LayerNorm(embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        pos_info = x[..., :12]
        pos_info = jnp.reshape(pos_info, (64 * 12,))
        pos_info = self.preprocess(pos_info)
        pos_info = jnp.reshape(pos_info, (64, self.dense_size))
        x = jnp.concat((x, pos_info), axis=1)

        # Square embedding.
        x = self.square_embedding(x)
        x = get_activation(self.default_activation)(x)
        x = self.norm(x)
        x = self.ma_gating(x)
        ffn_out = self.ffn(x)
        x = self.out_norm(x + ffn_out * self.alpha)
        return x


class Ffn(nnx.Module):
    def __init__(
        self,
        in_features: int,
        layer1_features: int,
        layer2_features: int,
        layer1_activation: net_pb2.NetworkFormat.ActivationFunction,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(
            in_features=in_features,
            out_features=layer1_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.activation = layer1_activation
        self.linear2 = nnx.Linear(
            in_features=layer1_features,
            out_features=layer2_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = get_activation(self.activation)(x)
        x = self.linear2(x)
        return x


class MaGating(nnx.Module):
    def __init__(self, feature_shape: tuple[int, ...], *, rngs: nnx.Rngs):
        self.mult_gate = Gating(
            feature_shape=feature_shape, additive=False, rngs=rngs
        )
        self.add_gate = Gating(
            feature_shape=feature_shape, additive=True, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.mult_gate(x)
        x = self.add_gate(x)
        return x


class Gating(nnx.Module):
    def __init__(
        self,
        feature_shape: tuple[int, ...],
        additive: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.additive = additive
        init_val = 0.0 if self.additive else 1.0
        self.gate = nnx.Param(
            jnp.full(feature_shape, init_val, dtype=jnp.float32)
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.additive:
            return inputs + self.gate.value
        else:
            effective_gate = jax.nn.relu(self.gate.value)
            return inputs * effective_gate


def scaled_xavier_normal(beta: float) -> Initializer:
    xavier_normal_fn = flax_initializers.glorot_normal()

    def init_fn(
        key: jax.Array, shape: Shape, dtype: jnp.dtype = jnp.float32
    ) -> jax.Array:
        weights = xavier_normal_fn(key, shape, dtype)
        return weights * jnp.sqrt(beta)

    return init_fn
