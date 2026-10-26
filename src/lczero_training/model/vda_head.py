"""Depth-attending value head ("vda").

A value head that reads the trunk's residual stream at every depth instead of
only at the last encoder block. The classic ValueHead sees one tensor -- the
final layer -- squeezed through a 64*num_channels -> 128 bottleneck; this head
adds two paths around that:

  per-square depth attention
      Each of the 64 squares attends over its OWN depth history: query from
      that square's final-layer vector, keys/values from the same square at
      every depth. The result is added to the head's per-square features
      BEFORE the bottleneck, so the bottleneck now compresses a depth-selected
      mixture rather than the last layer alone.

  global depth channel
      A board-level query (from the final layer's square-mean) attends over
      the square-mean position summary of every depth. Its output stays at
      full residual width and enters AFTER the bottleneck, so this path never
      passes through the 64*C -> hidden funnel at all.

  deep supervision (optional)
      A per-depth linear WDL probe on each depth's position summary. With a
      trainable trunk these push a value gradient into every encoder block.
      With a FROZEN trunk -- the graft experiment this was written for -- they
      train only the probes and read out how much value signal is linearly
      available at each depth.

Turning both channels off yields a lean last-layer-only value head, which is
the matched control this experiment compares against: same width, same
optimizer, same number of steps, minus the depth reading.

Everything is per position (no batch dimension); the caller vmaps.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .utils import get_activation, get_dtype

# WDL logits, error output, categorical output -- the ModelPrediction.value
# tuple shape. This head produces no error/categorical outputs.
ValueOutput = Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]


def _zeros_init(shape) -> jax.Array:
    return jnp.zeros(shape, jnp.float32)


class VdaValueHead(nnx.Module):
    """See module docstring. `num_taps` = encoder.num_blocks + 1."""

    def __init__(
        self,
        in_features: int,
        num_taps: int,
        config: model_config_pb2.VdaValueHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        rngs: nnx.Rngs,
    ):
        if num_taps < 2:
            raise ValueError("vda head needs at least two residual taps")
        self.activation = defaults.activation
        self.num_taps = num_taps
        self.channels = config.num_channels
        self.attention_dim = config.attention_dim
        self.hidden_size = config.hidden_size
        self.use_per_square = config.per_square_depth_attention
        self.use_global = config.global_depth_channel
        self.deep_supervision = config.deep_supervision
        cdt = get_dtype(defaults.compute_dtype)

        # Per-square features entering the bottleneck. `embed` is the classic
        # value head's input projection, read from the final layer only.
        self.embed = nnx.Linear(
            in_features=in_features,
            out_features=self.channels,
            dtype=cdt,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=self.channels * 64,
            out_features=self.hidden_size,
            dtype=cdt,
            rngs=rngs,
        )

        self.sq_query: Optional[nnx.Linear]
        self.sq_key: Optional[nnx.Linear]
        self.sq_value: Optional[nnx.Linear]
        self.sq_depth_key: Optional[nnx.Param]
        if self.use_per_square:
            self.sq_query = nnx.Linear(
                in_features=in_features,
                out_features=self.attention_dim,
                dtype=cdt,
                rngs=rngs,
            )
            self.sq_key = nnx.Linear(
                in_features=in_features,
                out_features=self.attention_dim,
                dtype=cdt,
                rngs=rngs,
            )
            self.sq_value = nnx.Linear(
                in_features=in_features,
                out_features=self.channels,
                dtype=cdt,
                rngs=rngs,
            )
            # Learned per-depth key offset, kept in float32 like q_scale. Zero
            # init means depth selection starts purely content-driven (the
            # taps already differ per depth, so this is not a symmetric init)
            # and any growth is the head learning a depth preference.
            self.sq_depth_key = nnx.Param(
                _zeros_init((num_taps, 1, self.attention_dim))
            )
        else:
            self.sq_query = None
            self.sq_key = None
            self.sq_value = None
            self.sq_depth_key = None

        self.gl_query: Optional[nnx.Linear]
        self.gl_key: Optional[nnx.Linear]
        self.gl_depth_key: Optional[nnx.Param]
        self.gl_out: Optional[nnx.Linear]
        if self.use_global:
            self.gl_query = nnx.Linear(
                in_features=in_features,
                out_features=self.attention_dim,
                dtype=cdt,
                rngs=rngs,
            )
            self.gl_key = nnx.Linear(
                in_features=in_features,
                out_features=self.attention_dim,
                dtype=cdt,
                rngs=rngs,
            )
            self.gl_depth_key = nnx.Param(
                _zeros_init((num_taps, self.attention_dim))
            )
            # Full residual width in, hidden width out: the bypass.
            self.gl_out = nnx.Linear(
                in_features=in_features,
                out_features=self.hidden_size,
                dtype=cdt,
                rngs=rngs,
            )
        else:
            self.gl_query = None
            self.gl_key = None
            self.gl_depth_key = None
            self.gl_out = None

        self.wdl = nnx.Linear(
            in_features=self.hidden_size,
            out_features=3,
            dtype=cdt,
            rngs=rngs,
        )

        self.aux: Optional[nnx.Dict]
        if self.deep_supervision:
            self.aux = nnx.Dict(
                {
                    aux_head_suffix(i): nnx.Linear(
                        in_features=in_features,
                        out_features=3,
                        dtype=cdt,
                        rngs=rngs,
                    )
                    for i in range(num_taps)
                }
            )
        else:
            self.aux = None

    def __call__(
        self, taps: list[jax.Array]
    ) -> Tuple[ValueOutput, list[jax.Array]]:
        assert len(taps) == self.num_taps, (
            f"vda head expects {self.num_taps} taps, got {len(taps)}"
        )
        final = taps[-1]
        stacked = jnp.stack(taps, axis=0)  # (taps, 64, features)
        # Square-mean position summary per depth; also the global channel's
        # value tensor, kept at full residual width.
        summary = jnp.mean(stacked, axis=1)  # (taps, features)

        features = self.embed(final)  # (64, channels)
        if self.use_per_square:
            assert self.sq_value is not None
            weights = self._per_square_weights(stacked, final)  # (taps, 64)
            values = self.sq_value(stacked)  # (taps, 64, channels)
            features = features + jnp.einsum(
                "ts,tsc->sc", weights.astype(values.dtype), values
            )

        hidden = self.dense1(
            get_activation(self.activation)(features.flatten())
        )

        if self.use_global:
            assert self.gl_out is not None
            gl_weights = self._global_weights(summary)  # (taps,)
            mixed = jnp.einsum(
                "t,td->d", gl_weights.astype(summary.dtype), summary
            )
            hidden = hidden + self.gl_out(mixed)

        hidden = get_activation(self.activation)(hidden)
        wdl = self.wdl(hidden).astype(jnp.float32)

        aux_logits: list[jax.Array] = []
        if self.aux is not None:
            aux_logits = [
                self.aux[aux_head_suffix(i)](summary[i]).astype(jnp.float32)
                for i in range(self.num_taps)
            ]
        return (wdl, None, None), aux_logits

    def depth_attention(
        self, taps: list[jax.Array]
    ) -> dict[str, Optional[jax.Array]]:
        """Both channels' attention distributions over depth, for analysis.

        Returns float32 `per_square` (taps, 64) and `global` (taps,), each
        summing to 1 along the depth axis. Not used by the forward pass; this
        is what answers "which depths does the head actually read".
        """
        final = taps[-1]
        stacked = jnp.stack(taps, axis=0)
        return {
            "per_square": (
                self._per_square_weights(stacked, final)
                if self.use_per_square
                else None
            ),
            "global": (
                self._global_weights(jnp.mean(stacked, axis=1))
                if self.use_global
                else None
            ),
        }

    def _per_square_weights(
        self, stacked: jax.Array, final: jax.Array
    ) -> jax.Array:
        assert self.sq_query is not None
        assert self.sq_key is not None
        assert self.sq_depth_key is not None
        query = self.sq_query(final)  # (64, attention_dim)
        keys = self.sq_key(stacked) + self.sq_depth_key.value.astype(
            stacked.dtype
        )  # (taps, 64, attention_dim)
        scores = jnp.einsum("sd,tsd->ts", query, keys)
        # Softmax over depth in float32, for the same reason the attention
        # softmax is: a bf16 softmax over a handful of near-equal logits loses
        # most of the distinction between depths.
        return nnx.softmax(scores.astype(jnp.float32) / self._scale(), axis=0)

    def _global_weights(self, summary: jax.Array) -> jax.Array:
        assert self.gl_query is not None
        assert self.gl_key is not None
        assert self.gl_depth_key is not None
        query = self.gl_query(summary[-1])  # (attention_dim,)
        keys = self.gl_key(summary) + self.gl_depth_key.value.astype(
            summary.dtype
        )  # (taps, attention_dim)
        scores = jnp.einsum("td,d->t", keys, query)
        return nnx.softmax(scores.astype(jnp.float32) / self._scale(), axis=0)

    def _scale(self) -> jax.Array:
        return jnp.sqrt(jnp.asarray(self.attention_dim, jnp.float32))


def aux_head_suffix(depth_index: int) -> str:
    """Name suffix of the deep-supervision probe at `depth_index`.

    Depth 0 is the embedding output, depth i>0 the output of encoder block
    i-1. Zero padded so head names sort by depth.
    """
    return f"aux{depth_index:02d}"


def aux_head_name(head_name: str, depth_index: int) -> str:
    """Value-head name the probe at `depth_index` is published under."""
    return f"{head_name}_{aux_head_suffix(depth_index)}"
