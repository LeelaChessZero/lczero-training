import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

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
        deepnorm_beta = math.pow(8.0 * config.encoder.num_blocks, -0.25)

        self.embedding = Embedding(
            input_channels=self._input_channels,
            config=config.embedding,
            defaults=config.defaults,
            deepnorm_alpha=math.pow(2.0 * config.encoder.num_blocks, -0.25),
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        assert self.config.encoder.num_blocks > 0

        self.encoders = EncoderTower(
            in_features=config.embedding.embedding_size,
            config=config.encoder,
            defaults=config.defaults,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        self.value_heads = {
            head_config.name: ValueHead(
                in_features=config.embedding.embedding_size,
                config=head_config,
                defaults=config.defaults,
                rngs=rngs,
            )
            for head_config in config.value_head
        }

        policy_shared_embedding = None
        if config.HasField("shared_policy_heads_embedding_size"):
            policy_shared_embedding = nnx.Linear(
                in_features=config.embedding.embedding_size,
                out_features=config.shared_policy_heads_embedding_size,
                rngs=rngs,
            )

        self.policy_heads = {
            head_config.name: PolicyHead(
                in_features=config.embedding.embedding_size,
                config=head_config,
                defaults=config.defaults,
                shared_embedding=policy_shared_embedding,
                rngs=rngs,
            )
            for head_config in config.policy_head
        }
        self.movesleft_heads = {
            head_config.name: MovesLeftHead(
                in_features=config.embedding.embedding_size,
                config=head_config,
                defaults=config.defaults,
                rngs=rngs,
            )
            for head_config in config.movesleft_head
        }

    def __call__(
        self, x: jax.Array
    ) -> Tuple[
        dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]
    ]:
        x = jnp.astype(x, get_dtype(self.config.defaults.compute_dtype))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))
        x = self.embedding(x)
        x = self.encoders(x)

        value = {name: head(x) for name, head in self.value_heads.items()}
        policy = {name: head(x) for name, head in self.policy_heads.items()}
        movesleft = {
            name: head(x) for name, head in self.movesleft_heads.items()
        }

        return value, policy, movesleft
