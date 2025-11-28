from typing import Tuple

import jax
from flax import nnx

from proto import model_config_pb2

from .utils import get_activation


class ValueHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.ValueHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        rngs: nnx.Rngs,
    ):
        self.activation = defaults.activation
        self.has_error_output = config.has_error_output
        self.embed = nnx.Linear(
            in_features=in_features,
            out_features=config.num_channels,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=config.num_channels * 64,
            out_features=128,
            rngs=rngs,
        )
        self.wdl = nnx.Linear(
            in_features=128,
            out_features=3,
            rngs=rngs,
        )
        if self.has_error_output:
            self.error = nnx.Linear(
                in_features=128,
                out_features=1,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, ...]:
        x = self.embed(x).flatten()
        x = get_activation(self.activation)(x)
        x = self.dense1(x)
        x = get_activation(self.activation)(x)
        wdl = self.wdl(x)

        if self.has_error_output:
            error = nnx.sigmoid(self.error(x))
            return (wdl, error)
        return (wdl,)

    def predict(self, x: jax.Array) -> jax.Array:
        return nnx.softmax(self(x)[0])
