import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .embedding import Embedding
from .encoder import EncoderTower
from .movesleft_head import MovesLeftHead
from .pda_head import PdaPolicyHead, probe_head_name
from .policy_head import PolicyHead
from .utils import get_dtype
from .value_head import ValueHead
from .vda_head import VdaValueHead, aux_head_name


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ModelPrediction:
    """Output predictions from LczeroModel.

    Fields:
        value: Dictionary mapping head names to value prediction tuples.
        policy: Dictionary mapping head names to policy logits.
        movesleft: Dictionary mapping head names to moves-left predictions.
    """

    value: dict[str, Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]]
    policy: dict[str, jax.Array]
    movesleft: dict[str, jax.Array]


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

        self.value_heads = nnx.Dict(
            {
                head_config.name: ValueHead(
                    in_features=config.embedding.embedding_size,
                    config=head_config,
                    defaults=config.defaults,
                    rngs=rngs,
                )
                for head_config in config.value_head
            }
        )

        # Named to appear before 'policy_heads' alphabetically in pytree state.
        # This ensures shared embedding appears at parent level during
        # serialization.
        self.policy_embedding_shared: Optional[nnx.Linear]
        if config.HasField("shared_policy_embedding_size"):
            self.policy_embedding_shared = nnx.Linear(
                in_features=config.embedding.embedding_size,
                out_features=config.shared_policy_embedding_size,
                rngs=rngs,
            )
        else:
            self.policy_embedding_shared = None

        self.policy_heads = nnx.Dict(
            {
                head_config.name: PolicyHead(
                    in_features=config.embedding.embedding_size,
                    config=head_config,
                    defaults=config.defaults,
                    shared_embedding=self.policy_embedding_shared,
                    rngs=rngs,
                )
                for head_config in config.policy_head
            }
        )
        self.movesleft_heads = nnx.Dict(
            {
                head_config.name: MovesLeftHead(
                    in_features=config.embedding.embedding_size,
                    config=head_config,
                    defaults=config.defaults,
                    rngs=rngs,
                )
                for head_config in config.movesleft_head
            }
        )

        # One tap per residual-stream depth: the embedding output plus each
        # encoder block's output.
        self._num_taps = config.encoder.num_blocks + 1
        self.vda_value_heads = nnx.Dict(
            {
                head_config.name: VdaValueHead(
                    in_features=config.embedding.embedding_size,
                    num_taps=self._num_taps,
                    config=head_config,
                    defaults=config.defaults,
                    rngs=rngs,
                )
                for head_config in config.vda_value_head
            }
        )
        self._has_vda_heads = len(config.vda_value_head) > 0

        # Depth-attending policy heads read the same taps. The plumbing is
        # shared: whichever of the two families is present turns the tap list
        # on, and both read the same `taps`.
        self.pda_policy_heads = nnx.Dict(
            {
                head_config.name: PdaPolicyHead(
                    in_features=config.embedding.embedding_size,
                    num_taps=self._num_taps,
                    config=head_config,
                    defaults=config.defaults,
                    rngs=rngs,
                )
                for head_config in config.pda_policy_head
            }
        )
        self._has_pda_heads = len(config.pda_policy_head) > 0

    def __call__(self, x: jax.Array) -> ModelPrediction:
        x = jnp.astype(x, get_dtype(self.config.defaults.compute_dtype))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))
        x = self.embedding(x)
        taps: list[jax.Array] | None = (
            [] if (self._has_vda_heads or self._has_pda_heads) else None
        )
        x = self.encoders(x, tap_out=taps)

        value = {name: head(x) for name, head in self.value_heads.items()}
        policy = {name: head(x) for name, head in self.policy_heads.items()}
        movesleft = {
            name: head(x) for name, head in self.movesleft_heads.items()
        }

        if taps is not None:
            # vda outputs join the ordinary value dict, so value losses and
            # value metrics address them by name like any other value head.
            # Deep-supervision probes are published as one value head per
            # depth, which gives a per-depth value curve for free.
            for name, head in self.vda_value_heads.items():
                prediction, aux_logits = head(taps)
                value[name] = prediction
                for depth, logits in enumerate(aux_logits):
                    value[aux_head_name(name, depth)] = (logits, None, None)
            # pda outputs join the ordinary policy dict for the same reason:
            # policy losses and policy metrics address them by name like any
            # other policy head, and each per-depth probe becomes its own
            # policy head, which gives a per-depth policy curve for free.
            for name, head in self.pda_policy_heads.items():
                logits, probe_logits = head(taps)
                policy[name] = logits
                for depth, probe in enumerate(probe_logits):
                    policy[probe_head_name(name, depth)] = probe

        return ModelPrediction(value=value, policy=policy, movesleft=movesleft)
