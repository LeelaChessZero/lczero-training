import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .shared import Ffn
from .utils import get_activation


class Embedding(nnx.Module):
    """Computes embeddings for the input features."""

    def __init__(
        self,
        *,
        input_channels: int,
        config: model_config_pb2.EmbeddingConfig,
        defaults: model_config_pb2.DefaultsConfig,
        rngs: nnx.Rngs,
    ):
        self._input_channels = input_channels
        dense_size = config.dense_size
        embedding_size = config.embedding_size
        self.activation = defaults.activation

        assert dense_size > 0
        self.preprocess = nnx.Linear(
            in_features=64 * 12,
            out_features=64 * dense_size,
            rngs=rngs,
        )

        assert embedding_size > 0
        self.embedding = nnx.Linear(
            in_features=input_channels + dense_size,
            out_features=embedding_size,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embedding_size, rngs=rngs)
        self.ma_gating = MaGating(feature_shape=(64, embedding_size), rngs=rngs)
        self.ffn = Ffn(
            in_features=embedding_size,
            hidden_features=config.dff,
            hidden_activation=defaults.ffn_activation,
            rngs=rngs,
        )
        self.out_norm = nnx.LayerNorm(embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Preprocess positional info and concatenate to input.
        pos_info = self.preprocess(x[..., :12].flatten()).reshape((64, -1))
        x = jnp.concatenate([x, pos_info], axis=1)

        # Square embedding.
        x = self.embedding(x)
        x = get_activation(self.activation)(x)
        x = self.norm(x)
        x = self.ma_gating(x)

        # FFN block with residual connection and layer norm.
        x = self.out_norm(x + self.ffn(x))
        return x


class MaGating(nnx.Module):
    """Applies multiplicative and additive gating."""

    def __init__(self, feature_shape: tuple[int, ...], *, rngs: nnx.Rngs):
        self.mult_gate = Gating(
            feature_shape=feature_shape, additive=False, rngs=rngs
        )
        self.add_gate = Gating(
            feature_shape=feature_shape, additive=True, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.add_gate(self.mult_gate(x))


class Gating(nnx.Module):
    def __init__(
        self,
        feature_shape: tuple[int, ...],
        additive: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.additive = additive
        init_val = 0.0 if self.additive else 1.0
        self.gate = nnx.Param(
            jnp.full(feature_shape, init_val, dtype=jnp.float32)
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.additive:
            return inputs + self.gate.value

        effective_gate = jax.nn.relu(self.gate.value)
        return inputs * effective_gate
