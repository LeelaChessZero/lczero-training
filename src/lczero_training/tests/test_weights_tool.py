"""Test weights tool arithmetic operations."""

import numpy as np

import proto.net_pb2 as net_pb2
from lczero_training.tools.weight_wrappers import NetWrapper


def test_weights_arithmetic() -> None:
    """Test arithmetic operations on simple networks."""
    # Create network A with a single layer containing value 10.0.
    net_a = net_pb2.Net()
    net_a.format.weights_encoding = net_pb2.Format.LINEAR16
    net_a.weights.ip1_val_w.min_val = 10.0
    net_a.weights.ip1_val_w.max_val = 10.0
    # LINEAR16 encoding: value 10.0 maps to uint16 value 32767 (mid-point).
    net_a.weights.ip1_val_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Create network B with a single layer containing value 20.0.
    net_b = net_pb2.Net()
    net_b.format.weights_encoding = net_pb2.Format.LINEAR16
    net_b.weights.ip1_val_w.min_val = 20.0
    net_b.weights.ip1_val_w.max_val = 20.0
    net_b.weights.ip1_val_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Wrap the networks.
    wrapper_a = NetWrapper(net_a)
    wrapper_b = NetWrapper(net_b)

    # Compute: output = 0.2*A + 0.8*B
    # Expected: 0.2*10 + 0.8*20 = 2 + 16 = 18
    output = 0.2 * wrapper_a + 0.8 * wrapper_b

    # Check the result.
    result_value = output.weights.ip1_val_w.value
    expected = 18.0

    assert result_value.shape == (1,), (
        f"Expected shape (1,), got {result_value.shape}"
    )
    assert np.isclose(result_value[0], expected, rtol=1e-4), (
        f"Expected {expected}, got {result_value[0]}"
    )
