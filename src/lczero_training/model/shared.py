from typing import Callable

import jax
from flax import nnx

from proto import net_pb2

from .utils import get_activation


class Ffn(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_activation: net_pb2.NetworkFormat.ActivationFunction,
        kernel_init: Callable[..., jax.Array],
        *,
        rngs: nnx.Rngs,
    ):
        out_features = in_features
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
        return x
