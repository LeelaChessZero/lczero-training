import dataclasses
import logging
from typing import Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lczero_training.convert.leela_pytree_visitor import (
    LeelaPytreeWeightsVisitor,
)
from proto import net_pb2

logger = logging.getLogger(__name__)


class JaxToLeela(LeelaPytreeWeightsVisitor):
    def tensor(
        self,
        param: nnx.Param,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        weights = param.value.T.flatten().astype(jnp.float32)
        min_val, max_val = jnp.min(weights), jnp.max(weights)
        range_val = max_val - min_val

        # Normalize to [0, 1], handling the case where all weights are equal.
        normalized = cast(
            jax.Array,
            jnp.where(range_val > 1e-8, (weights - min_val) / range_val, 0.5),
        )

        # Scale to uint16 and convert to bytes.
        quantized = jnp.round(normalized * 65535.0).astype(jnp.uint16)
        leela.params = np.asarray(quantized).tobytes()
        leela.min_val = float(min_val)
        leela.max_val = float(max_val)

        assert len(leela.params) // 2 == weights.size


@dataclasses.dataclass
class LeelaExportOptions:
    min_version: str
    license: Optional[str]


def jax_to_leela(
    jax_weights: nnx.State, export_options: LeelaExportOptions
) -> net_pb2.Net:
    lc0_weights = net_pb2.Net()
    lc0_weights.magic = 0x1C0
    if export_options.license:
        lc0_weights.license = export_options.license
    (
        lc0_weights.min_version.major,
        lc0_weights.min_version.minor,
        lc0_weights.min_version.patch,
    ) = _split_version(export_options.min_version)
    lc0_weights.format = _make_format()

    visitor = JaxToLeela(jax_weights, lc0_weights)
    visitor.run()

    return lc0_weights


def _split_version(version_str: str) -> tuple[int, int, int]:
    """Splits a version string like "v12.34.56" into (12, 34, 56)."""
    parts = (version_str.lstrip("v").split(".") + ["0", "0"])[:3]
    return cast(tuple[int, int, int], tuple(map(int, parts)))


def _make_format() -> net_pb2.Format:
    fmt = net_pb2.Format()
    fmt.weights_encoding = fmt.LINEAR16
    netfmt = fmt.network_format
    netfmt.input = netfmt.INPUT_CLASSICAL_112_PLANE
    netfmt.output = netfmt.OUTPUT_WDL
    netfmt.network = netfmt.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    netfmt.policy = netfmt.POLICY_ATTENTION
    netfmt.value = netfmt.VALUE_WDL
    netfmt.moves_left = netfmt.MOVES_LEFT_V1
    netfmt.default_activation = netfmt.DEFAULT_ACTIVATION_MISH
    netfmt.smolgen_activation = netfmt.ACTIVATION_SWISH
    netfmt.ffn_activation = netfmt.ACTIVATION_DEFAULT
    netfmt.input_embedding = netfmt.INPUT_EMBEDDING_PE_DENSE

    return fmt
