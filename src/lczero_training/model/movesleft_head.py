import jax
from flax import nnx

from proto import model_config_pb2

from .utils import get_activation


class MovesLeftHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.MovesLeftHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = defaults.activation
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
        self.out = nnx.Linear(
            in_features=128,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embed(x).flatten()
        x = get_activation(self.activation)(x)
        x = self.dense1(x)
        x = get_activation(self.activation)(x)
        x = self.out(x)
        return nnx.relu(x)
