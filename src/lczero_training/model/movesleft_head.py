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
        self.embedding = nnx.Linear(
            in_features=in_features,
            out_features=config.embedding_size,
            rngs=rngs,
        )

        self.fc4 = nnx.Linear(
            in_features=config.embedding_size * 64,
            out_features=128,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            in_features=128,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding(x).flatten()
        x = get_activation(self.activation)(x)
        x = self.fc4(x)
        x = get_activation(self.activation)(x)
        x = self.out(x)
        return nnx.relu(x)
