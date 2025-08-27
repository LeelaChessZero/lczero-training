import argparse
import gzip
import math

import jax.numpy as jnp
from flax import nnx, serialization

from lczero_training.model.model import LczeroModel
from proto import net_pb2

from .leela_pytree_visitor import LeelaPytreeWeightsVisitor
from .leela_to_modelconfig import leela_to_modelconfig


def _fix_older_weights_file(file: net_pb2.Net) -> None:
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
            print("Weights file has multihead format, updating format flag")
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
        values = values.reshape(param.shape)
        param.values = values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Leela Zero weights to JAX format and back."
    )
    parser.add_argument(
        "input", type=str, help="Path to the input Lc0 weights file."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        help="Output path to the ModelConfig textproto.",
    )
    parser.add_argument(
        "--weights_dtype",
        default="F32",
        type=str,
        help="The data type of the weights.",
    )
    parser.add_argument(
        "--compute_dtype",
        default="BF16",
        type=str,
        help="The data type for computation.",
    )
    parser.add_argument(
        "--jax_checkpoint",
        type=str,
        help="Path to save the output JAX checkpoint.",
    )

    args = parser.parse_args()

    lc0_weights = net_pb2.Net()
    with gzip.open(args.input, "rb") as f:
        contents = f.read()
        assert isinstance(contents, bytes)
        lc0_weights.ParseFromString(contents)

    _fix_older_weights_file(lc0_weights)

    config = leela_to_modelconfig(
        lc0_weights, args.weights_dtype, args.compute_dtype
    )
    if args.model_config:
        with open(args.model_config, "w") as f:
            f.write(str(config))

    if args.jax_checkpoint is None:
        return

    model = LczeroModel(config=config, rngs=nnx.Rngs(params=42))
    state = nnx.state(model)
    as_dict = nnx.to_pure_dict(state)
    visitor = LeelaToJax(as_dict, lc0_weights)
    visitor.run()

    with open(args.jax_checkpoint, "wb") as f:
        f.write(serialization.to_bytes(as_dict))


if __name__ == "__main__":
    main()
