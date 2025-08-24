import jax
import jax.numpy as jnp
import jax.random
from flax import nnx

from hlo_pb2 import XlaShapeProto
from proto import model_config_pb2, net_pb2

from .embedding import Embedding
from .encoder import EncoderTower
from .utils import get_dtype


class LczeroModel(nnx.Module):
    def __init__(self, config: model_config_pb2.ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self._input_channels = 112

        self.embedding = Embedding(
            input_channels=self._input_channels,
            config=config.embedding,
            rngs=rngs,
        )

        assert self.config.policy == net_pb2.NetworkFormat.POLICY_ATTENTION
        assert self.config.encoder.num_blocks > 0

        self.encoders = EncoderTower(
            in_features=config.embedding.embedding_size,
            config=config.encoder,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.astype(x, get_dtype(self.config.activations_type))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))
        x = self.embedding(x)
        x = self.encoders(x)

        return x


def _tmp_make_config() -> model_config_pb2.ModelConfig:
    config = model_config_pb2.ModelConfig()

    config.weights_type = XlaShapeProto.F32
    config.activations_type = XlaShapeProto.BF16
    config.policy = net_pb2.NetworkFormat.POLICY_ATTENTION

    config.embedding.dense_size = 512
    config.embedding.dff = 1536
    config.embedding.embedding_size = 1024
    config.embedding.activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.embedding.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_MISH

    config.encoder.num_blocks = 15
    config.encoder.dff = 1536
    config.encoder.d_model = 1024
    config.encoder.heads = 32
    config.encoder.activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.encoder.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_MISH

    config.encoder.smolgen.hidden_channels = 32
    config.encoder.smolgen.hidden_size = 256
    config.encoder.smolgen.gen_size = 256
    config.encoder.smolgen.activation = net_pb2.NetworkFormat.ACTIVATION_SWISH

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
    if isinstance(output, (list, tuple)):
        for o in output:
            print(o.shape)
    else:
        print(output.shape)
