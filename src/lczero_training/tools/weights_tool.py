"""Main API for loading and saving Lc0 weight files."""

import gzip

from proto import net_pb2

from .weight_wrappers import NetWrapper


def load_weights(path: str) -> NetWrapper:
    """Load Lc0 weights file (.pb or .pb.gz)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            contents = f.read()
    else:
        with open(path, "rb") as f:
            contents = f.read()

    net = net_pb2.Net()
    net.ParseFromString(contents)
    return NetWrapper(net)


def save_weights(
    wrapper: NetWrapper, path: str, encoding: str = "FLOAT16"
) -> None:
    """Save weights to file."""
    encoding_map = {
        "LINEAR16": net_pb2.Weights.Layer.LINEAR16,
        "FLOAT16": net_pb2.Weights.Layer.FLOAT16,
        "BFLOAT16": net_pb2.Weights.Layer.BFLOAT16,
    }
    enc_value = encoding_map[encoding.upper()]
    wrapper.save(path, enc_value)
