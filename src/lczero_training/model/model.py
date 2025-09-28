import math
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random
from flax import nnx

from proto import model_config_pb2, net_pb2
from proto.hlo_pb2 import XlaShapeProto

from .embedding import Embedding
from .encoder import EncoderTower
from .movesleft_head import MovesLeftHead
from .policy_head import PolicyHead
from .utils import get_dtype
from .value_head import ValueHead


class LczeroModel(nnx.Module):
    def __init__(self, config: model_config_pb2.ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self._input_channels = 112

        self.embedding = Embedding(
            input_channels=self._input_channels,
            config=config.embedding,
            defaults=config.defaults,
            alpha=math.pow(2.0 * config.encoder.num_blocks, -0.25),
            rngs=rngs,
        )

        assert self.config.encoder.num_blocks > 0

        self.encoders = EncoderTower(
            in_features=config.embedding.embedding_size,
            config=config.encoder,
            defaults=config.defaults,
            rngs=rngs,
        )

        self.value_head = ValueHead(
            in_features=config.embedding.embedding_size,
            config=config.value_head,
            defaults=config.defaults,
            rngs=rngs,
        )

        self.policy_head = PolicyHead(
            in_features=config.embedding.embedding_size,
            config=config.policy_head,
            defaults=config.defaults,
            rngs=rngs,
        )
        self.movesleft_head = MovesLeftHead(
            in_features=config.embedding.embedding_size,
            config=config.movesleft_head,
            defaults=config.defaults,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        x = jnp.astype(x, get_dtype(self.config.defaults.compute_dtype))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))
        x = self.embedding(x)
        x = self.encoders(x)

        value = self.value_head(x)
        policy = self.policy_head(x)
        movesleft = self.movesleft_head(x)

        return value, policy, movesleft


def _tmp_make_config() -> model_config_pb2.ModelConfig:
    config = model_config_pb2.ModelConfig()

    config.defaults.compute_dtype = XlaShapeProto.BF16
    config.defaults.activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.defaults.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_MISH

    config.embedding.dense_size = 512
    config.embedding.dff = 1536
    config.embedding.embedding_size = 1024

    config.encoder.num_blocks = 15
    config.encoder.dff = 1536
    config.encoder.d_model = 1024
    config.encoder.heads = 32

    config.encoder.smolgen.hidden_channels = 32
    config.encoder.smolgen.hidden_size = 256
    config.encoder.smolgen.gen_size = 256
    config.encoder.smolgen.activation = net_pb2.NetworkFormat.ACTIVATION_SWISH

    config.policy_head.embedding_size = 1024
    config.policy_head.d_model = 1024

    config.value_head.num_channels = 128

    config.movesleft_head.num_channels = 32

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
