import gzip
import logging
import math
from typing import Optional

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx, serialization

import hlo_pb2
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import JitTrainingState, TrainingState
from proto import net_pb2

from .leela_pytree_visitor import LeelaPytreeWeightsVisitor
from .leela_to_modelconfig import leela_to_modelconfig

logger = logging.getLogger(__name__)


def fix_older_weights_file(file: net_pb2.Net) -> None:
    nf = net_pb2.NetworkFormat
    has_network_format = file.format.HasField("network_format")
    network_format = (
        file.format.network_format.network if has_network_format else None
    )

    net = file.format.network_format

    if not has_network_format:
        # Older protobufs don't have format definition.
        net.input = nf.INPUT_CLASSICAL_112_PLANE
        net.output = nf.OUTPUT_CLASSICAL
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_CLASSICAL:
        # Populate policyFormat and valueFormat fields in old protobufs
        # without these fields.
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_SE:
        net.network = nf.NETWORK_SE_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif (
        network_format == nf.NETWORK_SE_WITH_HEADFORMAT
        and len(file.weights.encoder) > 0
    ):
        # Attention body network made with old protobuf.
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
        if file.weights.HasField("smolgen_w"):
            # Need to override activation defaults for smolgen.
            net.ffn_activation = nf.ACTIVATION_RELU_2
            net.smolgen_activation = nf.ACTIVATION_SWISH
    elif network_format == nf.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT:
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT

    if (
        file.format.network_format.network
        == nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    ):
        weights = file.weights
        if weights.HasField("policy_heads") and weights.HasField("value_heads"):
            logger.info(
                "Weights file has multihead format, updating format flag"
            )
            net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
            net.input_embedding = nf.INPUT_EMBEDDING_PE_DENSE
        if not file.format.network_format.HasField("input_embedding"):
            net.input_embedding = nf.INPUT_EMBEDDING_PE_MAP


class LeelaToJax(LeelaPytreeWeightsVisitor):
    def tensor(
        self,
        param: nnx.Param,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0

        values = jnp.frombuffer(leela.params, dtype=jnp.uint16)
        values = values.astype(jnp.float32)
        values /= 65535.0
        values *= leela.max_val - leela.min_val
        values += leela.min_val
        values = values.astype(param.dtype)
        values = values.reshape(param.shape[::-1]).transpose()
        param.value = values


def leela_to_jax(
    input_path: str,
    weights_dtype: str,
    compute_dtype: str,
    output_modelconfig: Optional[str],
    output_serialized_jax: Optional[str],
    output_orbax_checkpoint: Optional[str],
    print_modelconfig: bool = False,
) -> None:
    lc0_weights = net_pb2.Net()
    with gzip.open(input_path, "rb") as f:
        contents = f.read()
        assert isinstance(contents, bytes)
        lc0_weights.ParseFromString(contents)

    fix_older_weights_file(lc0_weights)

    config = leela_to_modelconfig(
        lc0_weights,
        getattr(hlo_pb2.XlaShapeProto, weights_dtype),
        getattr(hlo_pb2.XlaShapeProto, compute_dtype),
    )

    if print_modelconfig:
        print(config)

    if output_modelconfig:
        with open(output_modelconfig, "w") as f:
            f.write(str(config))

    if output_serialized_jax is None and output_orbax_checkpoint is None:
        return

    model = LczeroModel(config=config, rngs=nnx.Rngs(params=42))
    state = nnx.state(model)
    visitor = LeelaToJax(state, lc0_weights)
    visitor.run()

    if output_serialized_jax:
        with open(output_serialized_jax, "wb") as f:
            f.write(serialization.to_bytes(state))

    if output_orbax_checkpoint:
        jit_state = JitTrainingState(
            step=lc0_weights.training_params.training_steps,
            model_state=state,
            opt_state=None,
        )
        training_state = TrainingState(
            jit_state=jit_state,
        )
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(output_orbax_checkpoint, training_state)
        checkpointer.wait_until_finished()

    # nnx.update(model, state)
