import jax
import jax.numpy as jnp
import jax.random
from flax import nnx

from hlo_pb2 import XlaShapeProto
from proto import net_pb2
from proto.model_config_pb2 import ModelConfig

from .embedding import Embedding
from .utils import get_dtype


class LczeroModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self._input_channels = 112
        self._rngs = rngs

        self.embedding = Embedding(
            input_channels=self._input_channels,
            dense_size=config.embedding_dense_sz,
            embedding_size=config.embedding_size,
            dff=config.encoder_dff,
            default_activation=config.default_activation,
            ffn_activation=config.ffn_activation,
            rngs=self._rngs,
        )

        assert self.config.policy == net_pb2.NetworkFormat.POLICY_ATTENTION
        assert self.config.encoder_layers > 0

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.astype(x, get_dtype(self.config.activations_type))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))

        x = self.embedding(x)
        return x


def _tmp_make_config() -> ModelConfig:
    config = ModelConfig()
    config.encoder_layers = 15
    config.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    config.default_activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.weights_type = XlaShapeProto.F32
    config.activations_type = XlaShapeProto.BF16
    config.embedding_dense_sz = 512
    config.embedding_size = 1024
    config.encoder_dff = 1536
    return config


if __name__ == "__main__":
    rngs = nnx.Rngs(params=42)
    model = LczeroModel(config=_tmp_make_config(), rngs=rngs)
    # batch_model = nnx.vmap(model, in_axes=0, out_axes=0)
    print(model)

    key = jax.random.key(0)
    random_input = jax.random.normal(key, (112, 8, 8))
    output = model(random_input)
    print(output)
    print(output.shape)
