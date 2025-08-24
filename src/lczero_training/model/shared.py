import math

import jax
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import net_pb2

from .utils import get_activation


class Ffn(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_activation: net_pb2.NetworkFormat.ActivationFunction,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = in_features
        self.alpha = math.pow(2.0 * out_features, -0.25)
        beta = math.pow(8.0 * out_features, -0.25)
        kernel_init = flax_initializers.variance_scaling(
            scale=beta, mode="fan_avg", distribution="truncated_normal"
        )

        self.linear1 = nnx.Linear(
            in_features=in_features,
            out_features=hidden_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.activation = hidden_activation
        self.linear2 = nnx.Linear(
            in_features=hidden_features,
            out_features=out_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = get_activation(self.activation)(x)
        x = self.linear2(x)
        return x * self.alpha
