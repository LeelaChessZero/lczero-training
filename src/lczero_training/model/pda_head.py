"""Depth-attending policy head ("pda"), the policy analogue of vda_head.py.

The vanilla attention-policy head sees one tensor -- the last encoder block's
output -- projects it to per-square tokens and scores every (from, to) pair
with a q/k dot product plus a promotion offset. This head keeps that entire
output stage and changes only what feeds it:

  per-square depth attention
      Each of the 64 squares attends over its OWN depth history: query from
      that square's final-layer vector, keys from the same square at every
      depth. The depth-selected mixture is added to the head's per-square
      token features, so the q/k scoring reads a depth-enriched square rather
      than the last layer alone. This is exactly vda's per-square channel with
      the value head's bottleneck replaced by the policy head's q/k stage.

  deep supervision (optional)
      A minimal attention-policy probe (q/k + promotion, no token projection,
      no activation) on each depth's residual stream. Each probe emits full
      1858-way policy logits, so the per-depth policy accuracy curve reads out
      directly as `policy/<name>_aux<NN>/accuracy` -- the policy analogue of
      vda's per-depth WDL probe curve. With a trainable trunk these also push a
      policy gradient into every encoder block.

Turning the depth channel off yields the matched control: the SAME modules,
the SAME parameter count and the SAME op sequence, with the depth axis filled
by copies of the final layer instead of the real depth history. See
`PdaPolicyHead.__call__` for what that costs.

vda's second channel (the board-level "global depth channel") has no analogue
here on purpose. It existed because the value head funnels 64*C features
through a 64*C -> 128 bottleneck and the global channel was the path that
bypassed the funnel. The policy head has no bottleneck -- it stays per-square
all the way to the q/k product -- so there is nothing to bypass.

Everything is per position (no batch dimension); the caller vmaps.
"""

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .policy_head import _policy_map
from .utils import get_activation, get_dtype


def _zeros_init(shape) -> jax.Array:
    return jnp.zeros(shape, jnp.float32)


def _attention_policy_logits(
    q: jax.Array,
    k: jax.Array,
    promotion_dense: nnx.Linear,
    dk: float,
) -> jax.Array:
    """The vanilla policy head's output stage, verbatim.

    `q`/`k` are (64, d_model) per-square projections. Returns the 1858 policy
    logits in lc0 move order, in float32. Lifted out of PolicyHead.__call__
    unchanged so that this head and its probes are the same architecture class
    as the production head rather than a re-derivation of it.
    """
    qk = jnp.einsum("qd,kd->qk", q, k)

    promotion_keys = k[-8:, :]
    promotion_offsets = promotion_dense(promotion_keys)
    promotion_offsets = promotion_offsets.transpose((1, 0)) * dk
    # knight offset is added to the other three
    promotion_offsets = promotion_offsets[:3, :] + promotion_offsets[3:4, :]

    n_promo_logits = qk[-16:-8, -8:]
    q_promo_logits = jnp.expand_dims(
        n_promo_logits + promotion_offsets[0:1, :], axis=-1
    )
    r_promo_logits = jnp.expand_dims(
        n_promo_logits + promotion_offsets[1:2, :], axis=-1
    )
    b_promo_logits = jnp.expand_dims(
        n_promo_logits + promotion_offsets[2:3, :], axis=-1
    )
    promotion_logits = jnp.concatenate(
        [q_promo_logits, r_promo_logits, b_promo_logits], axis=-1
    )
    policy_attn_logits = qk / dk
    promotion_logits = promotion_logits.reshape((8, 24)) / dk

    logits = jnp.concatenate(
        [policy_attn_logits.flatten(), promotion_logits.flatten()], axis=-1
    )
    return logits[_policy_map].astype(jnp.float32)


class PdaPolicyProbe(nnx.Module):
    """Smallest attention-policy readout of one residual-stream depth.

    q/k straight off the tap with no token projection and no activation, then
    the production head's scoring stage. "Linear probe" in the same sense as
    vda's per-depth WDL probe: the only nonlinearity is the one the policy
    parameterization itself imposes (the q.k product), and there is no hidden
    layer to fit around a trunk that has not made the answer legible.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.q = nnx.Linear(
            in_features=in_features,
            out_features=d_model,
            dtype=dtype,
            rngs=rngs,
        )
        self.k = nnx.Linear(
            in_features=in_features,
            out_features=d_model,
            dtype=dtype,
            rngs=rngs,
        )
        self.promotion_dense = nnx.Linear(
            in_features=d_model,
            out_features=4,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.dk = math.sqrt(d_model)

    def __call__(self, tap: jax.Array) -> jax.Array:
        return _attention_policy_logits(
            self.q(tap), self.k(tap), self.promotion_dense, self.dk
        )


class PdaPolicyHead(nnx.Module):
    """See module docstring. `num_taps` = encoder.num_blocks + 1."""

    def __init__(
        self,
        in_features: int,
        num_taps: int,
        config: model_config_pb2.PdaPolicyHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        rngs: nnx.Rngs,
    ):
        if num_taps < 2:
            raise ValueError("pda head needs at least two residual taps")
        self.activation = defaults.activation
        self.num_taps = num_taps
        self.embedding_size = config.embedding_size
        self.d_model = config.d_model
        self.attention_dim = config.attention_dim
        self.use_per_square = config.per_square_depth_attention
        self.deep_supervision = config.deep_supervision
        cdt = get_dtype(defaults.compute_dtype)

        # The production head's token projection, read from the final layer
        # only. Deliberately NOT the shared policy embedding: a shared
        # projection would let this head's gradient reshape the tensor the
        # vanilla/soft/optimistic heads read, which is the one thing an added
        # head must not do to the arm it is being compared against.
        self.tokens = nnx.Linear(
            in_features=in_features,
            out_features=self.embedding_size,
            dtype=cdt,
            rngs=rngs,
        )

        # Depth attention, mirroring VdaValueHead's per-square channel.
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
            out_features=self.embedding_size,
            dtype=cdt,
            rngs=rngs,
        )
        # Learned per-depth key offset, float32 like vda's. Zero init means
        # depth selection starts purely content-driven (the taps already differ
        # per depth, so this is not a symmetric init).
        self.sq_depth_key = nnx.Param(
            _zeros_init((num_taps, 1, self.attention_dim))
        )

        self.q = nnx.Linear(
            in_features=self.embedding_size,
            out_features=self.d_model,
            dtype=cdt,
            rngs=rngs,
        )
        self.k = nnx.Linear(
            in_features=self.embedding_size,
            out_features=self.d_model,
            dtype=cdt,
            rngs=rngs,
        )
        self.promotion_dense = nnx.Linear(
            in_features=self.d_model,
            out_features=4,
            use_bias=False,
            dtype=cdt,
            rngs=rngs,
        )
        self.dk = math.sqrt(self.d_model)

        self.probes: Optional[nnx.Dict]
        if self.deep_supervision:
            self.probes = nnx.Dict(
                {
                    probe_suffix(i): PdaPolicyProbe(
                        in_features=in_features,
                        d_model=config.probe_d_model,
                        dtype=cdt,
                        rngs=rngs,
                    )
                    for i in range(num_taps)
                }
            )
        else:
            self.probes = None

    def __call__(
        self, taps: list[jax.Array]
    ) -> Tuple[jax.Array, list[jax.Array]]:
        assert len(taps) == self.num_taps, (
            f"pda head expects {self.num_taps} taps, got {len(taps)}"
        )
        final = taps[-1]
        stacked = self._depth_axis(taps)  # (taps, 64, features)

        features = self.tokens(final)  # (64, embedding_size)
        weights = self._per_square_weights(stacked, final)  # (taps, 64)
        # vda computes sq_value on every tap and then mixes; because sq_value
        # is one shared affine map and the depth weights sum to 1, mixing the
        # taps first and projecting once is the SAME tensor (W(sum w x) + b ==
        # sum w (W x + b)) for num_taps times fewer matmuls. Same head, same
        # parameters, same result.
        mixed = jnp.einsum("ts,tsf->sf", weights.astype(stacked.dtype), stacked)
        features = features + self.sq_value(mixed)

        features = get_activation(self.activation)(features)
        logits = _attention_policy_logits(
            self.q(features), self.k(features), self.promotion_dense, self.dk
        )

        probe_logits: list[jax.Array] = []
        if self.probes is not None:
            probe_logits = [
                self.probes[probe_suffix(i)](taps[i])
                for i in range(self.num_taps)
            ]
        return logits, probe_logits

    def depth_attention(self, taps: list[jax.Array]) -> jax.Array:
        """The attention distribution over depth, (taps, 64), for analysis.

        Sums to 1 along the depth axis. Not used by the forward pass; this is
        what answers "which depths does the policy head actually read".
        """
        return self._per_square_weights(self._depth_axis(taps), taps[-1])

    def _depth_axis(self, taps: list[jax.Array]) -> jax.Array:
        """What fills the depth axis: the real history, or the control's.

        The control (`per_square_depth_attention: false`) fills every depth
        slot with a copy of the FINAL layer, so it runs the identical module,
        the identical op sequence and the identical parameter count over an
        axis that carries no depth information. Two consequences, both
        intended and both worth stating:

          * the mixture collapses to `final` exactly (the weights sum to 1
            over identical vectors), so the control is a fresh policy head
            reading the last layer and nothing else -- the paired control this
            experiment needs;
          * `sq_query`/`sq_key`/`sq_depth_key` (16,800 of its 280,992
            parameters, 6%) therefore receive exactly zero gradient and stay
            at their init. The control's parameter BUDGET matches the
            treatment's; its trainable capacity is 94% of it. Any control
            reading only the final layer has this property -- there is no
            depth variation for a depth query to select over.
        """
        if self.use_per_square:
            return jnp.stack(taps, axis=0)
        final = taps[-1]
        return jnp.broadcast_to(final, (self.num_taps,) + final.shape)

    def _per_square_weights(
        self, stacked: jax.Array, final: jax.Array
    ) -> jax.Array:
        query = self.sq_query(final)  # (64, attention_dim)
        keys = self.sq_key(stacked) + self.sq_depth_key.value.astype(
            stacked.dtype
        )  # (taps, 64, attention_dim)
        scores = jnp.einsum("sd,tsd->ts", query, keys)
        # Softmax over depth in float32: a bf16 softmax over a handful of
        # near-equal logits loses most of the distinction between depths.
        return nnx.softmax(
            scores.astype(jnp.float32)
            / jnp.sqrt(jnp.asarray(self.attention_dim, jnp.float32)),
            axis=0,
        )


def probe_suffix(depth_index: int) -> str:
    """Name suffix of the deep-supervision probe at `depth_index`.

    Depth 0 is the embedding output, depth i>0 the output of encoder block
    i-1. Zero padded so head names sort by depth. Same convention as
    vda_head.aux_head_suffix, so the value and policy depth curves line up
    scalar-for-scalar in TensorBoard.
    """
    return f"aux{depth_index:02d}"


def probe_head_name(head_name: str, depth_index: int) -> str:
    """Policy-head name the probe at `depth_index` is published under."""
    return f"{head_name}_{probe_suffix(depth_index)}"
