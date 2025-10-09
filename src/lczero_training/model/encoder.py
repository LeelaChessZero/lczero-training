import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .shared import Ffn
from .utils import get_activation


class EncoderTower(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        deepnorm_init: Callable[..., jax.Array],
        rngs: nnx.Rngs,
    ):
        smolgen_shared_gen_dense = None
        assert config.HasField("smolgen")
        if config.HasField("smolgen"):
            smolgen_shared_gen_dense = nnx.Linear(
                in_features=config.smolgen.gen_size,
                out_features=64 * 64,
                use_bias=False,
                rngs=rngs,
            )

        self.encoders = nnx.Sequential(
            *[
                EncoderBlock(
                    in_features=in_features,
                    config=config,
                    defaults=defaults,
                    smol_gen_dense=smolgen_shared_gen_dense,
                    deepnorm_init=deepnorm_init,
                    rngs=rngs,
                )
                for _ in range(config.num_blocks)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.encoders(x)


class EncoderBlock(nnx.Module):
    """A single block of the transformer encoder."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: Optional[nnx.Linear],
        deepnorm_init: Callable[..., jax.Array],
        rngs: nnx.Rngs,
    ):
        assert (smol_gen_dense is not None) == config.HasField("smolgen")
        self.mha = MultiHeadAttention(
            in_features=in_features,
            config=config,
            defaults=defaults,
            smol_gen_dense=smol_gen_dense,
            deepnorm_init=deepnorm_init,
            rngs=rngs,
        )

        self.alpha = math.pow(2.0 * config.num_blocks, -0.25)
        self.ln1 = nnx.LayerNorm(in_features, epsilon=1e-3, rngs=rngs)
        self.ffn = Ffn(
            in_features=in_features,
            hidden_features=config.dff,
            hidden_activation=defaults.ffn_activation,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(in_features, epsilon=1e-3, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.mha(x) * self.alpha
        out1 = self.ln1(x)
        ffn_out = self.ffn(out1)
        return self.ln2(out1 + ffn_out * self.alpha)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: Optional[nnx.Linear],
        deepnorm_init: Callable[..., jax.Array],
        *,
        rngs: nnx.Rngs,
    ):
        depth = config.d_model
        assert depth % config.heads == 0, (
            "Model depth must be divisible by the number of heads."
        )
        self.activation = defaults.activation
        self.depth = depth
        self.num_heads = config.heads
        self.q = nnx.Linear(
            in_features=in_features, out_features=depth, rngs=rngs
        )
        self.k = nnx.Linear(
            in_features=in_features, out_features=depth, rngs=rngs
        )
        self.v = nnx.Linear(
            in_features=in_features,
            out_features=depth,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.output_dense = nnx.Linear(
            in_features=depth,
            out_features=in_features,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

        assert (smol_gen_dense is not None) == config.HasField("smolgen")
        self.smolgen = None
        if smol_gen_dense is not None:
            self.smolgen = Smolgen(
                in_features=in_features,
                config=config.smolgen,
                defaults=defaults,
                heads=config.heads,
                weight_gen_dense=smol_gen_dense,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        q, k, v = self.q(x), self.k(x), self.v(x)

        head_depth = self.depth // self.num_heads
        # Reshape for multi-head attention.
        q, k, v = (
            t.reshape((-1, self.num_heads, head_depth)).transpose((1, 0, 2))
            for t in (q, k, v)
        )

        # Scaled dot-product attention.
        logits = jnp.einsum("...qd,...kd->...qk", q, k)
        logits /= jnp.sqrt(k.shape[-1]).astype(k.dtype)

        if self.smolgen is not None:
            logits += self.smolgen(x)

        attention_weights = nnx.softmax(logits, axis=-1)
        scaled_attention = jnp.matmul(attention_weights, v)

        # Reshape back to original dimensions.
        scaled_attention = scaled_attention.transpose((1, 0, 2)).reshape(
            (-1, self.depth)
        )
        return self.output_dense(scaled_attention)


class Smolgen(nnx.Module):
    """Smolgen module for generating attention biases."""

    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.SmolgenConfig,
        defaults: model_config_pb2.DefaultsConfig,
        heads: int,
        weight_gen_dense: nnx.Linear,
        *,
        rngs: nnx.Rngs,
    ):
        self.heads = heads
        self.compress = nnx.Linear(
            in_features=in_features,
            out_features=config.hidden_channels,
            use_bias=False,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=config.hidden_channels * 64,
            out_features=config.hidden_size,
            rngs=rngs,
        )
        self.ln1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-3, rngs=rngs)

        self.dense2 = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.gen_size * heads,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(
            config.gen_size * heads, epsilon=1e-3, rngs=rngs
        )
        self.weight_gen_dense = weight_gen_dense
        self.activation = config.activation or defaults.activation

    def __call__(self, x: jax.Array) -> jax.Array:
        compressed = self.compress(x).flatten()
        hidden = self.dense1(compressed)
        hidden = get_activation(self.activation)(hidden)
        hidden = self.ln1(hidden)

        gen_from = self.dense2(hidden)
        gen_from = get_activation(self.activation)(gen_from)
        gen_from = self.ln2(gen_from)
        gen_from = gen_from.reshape((self.heads, -1))

        out = self.weight_gen_dense(gen_from)
        return out.reshape((self.heads, 64, 64))
