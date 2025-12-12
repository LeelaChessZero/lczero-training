"""Test weights tool arithmetic operations."""

import os
import tempfile

import numpy as np

import proto.net_pb2 as net_pb2
from lczero_training.tools.weight_wrappers import NetWrapper
from lczero_training.tools.weights_tool import load_weights, save_weights


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


def test_policy_head_replacement() -> None:
    """Test that assigning policy_heads actually replaces the data."""
    # Create network A with policy head value 10.0.
    net_a = net_pb2.Net()
    net_a.format.weights_encoding = net_pb2.Format.LINEAR16
    net_a.weights.policy_heads.ip_pol_w.min_val = 10.0
    net_a.weights.policy_heads.ip_pol_w.max_val = 10.0
    net_a.weights.policy_heads.ip_pol_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Create network B with policy head value 20.0.
    net_b = net_pb2.Net()
    net_b.format.weights_encoding = net_pb2.Format.LINEAR16
    net_b.weights.policy_heads.ip_pol_w.min_val = 20.0
    net_b.weights.policy_heads.ip_pol_w.max_val = 20.0
    net_b.weights.policy_heads.ip_pol_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Wrap and perform assignment.
    wrapper_a = NetWrapper(net_a)
    wrapper_b = NetWrapper(net_b)

    # Replace A's policy heads with B's.
    wrapper_a.weights.policy_heads = wrapper_b.weights.policy_heads

    # Verify in-memory replacement.
    result_value = wrapper_a.weights.policy_heads.ip_pol_w.value
    assert np.isclose(result_value[0], 20.0, rtol=1e-4), (
        f"Expected 20.0 (B's value), got {result_value[0]}"
    )

    # Verify persistence (round-trip).
    with tempfile.NamedTemporaryFile(suffix=".pb.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_weights(wrapper_a, tmp_path)
        reloaded = load_weights(tmp_path)
        reloaded_value = reloaded.weights.policy_heads.ip_pol_w.value
        assert np.isclose(reloaded_value[0], 20.0, rtol=1e-4), (
            f"After save/load: Expected 20.0, got {reloaded_value[0]}"
        )
    finally:
        os.unlink(tmp_path)


def test_policy_head_map_assignment() -> None:
    """Test assigning policy_heads when one has policy_head_map and another doesn't."""
    # Create network A with NO policy_head_map.
    net_a = net_pb2.Net()
    net_a.format.weights_encoding = net_pb2.Format.LINEAR16
    net_a.weights.policy_heads.ip_pol_w.min_val = 5.0
    net_a.weights.policy_heads.ip_pol_w.max_val = 5.0
    net_a.weights.policy_heads.ip_pol_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Create network B WITH policy_head_map.
    net_b = net_pb2.Net()
    net_b.format.weights_encoding = net_pb2.Format.LINEAR16
    # Add a policy_head_map entry with required key and value fields.
    policy_map = net_b.weights.policy_heads.policy_head_map.add()
    policy_map.key = "test_policy"
    policy_map.value.ip_pol_w.min_val = 15.0
    policy_map.value.ip_pol_w.max_val = 15.0
    policy_map.value.ip_pol_w.params = np.array(
        [32767], dtype=np.uint16
    ).tobytes()

    # Wrap networks.
    wrapper_a = NetWrapper(net_a)
    wrapper_b = NetWrapper(net_b)

    # Replace A's policy heads with B's (which has policy_head_map).
    wrapper_a.weights.policy_heads = wrapper_b.weights.policy_heads

    # Access policy_head_map through the cache to verify consistency.
    assert len(wrapper_a.weights.policy_heads.policy_head_map) == 1
    assert (
        wrapper_a.weights.policy_heads.policy_head_map[0]._proto.key
        == "test_policy"
    )

    # Verify can save without serialization errors.
    with tempfile.NamedTemporaryFile(suffix=".pb.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_weights(wrapper_a, tmp_path)
        # Verify round-trip preserves the policy_head_map.
        reloaded = load_weights(tmp_path)
        assert len(reloaded.weights.policy_heads.policy_head_map) == 1
        assert (
            reloaded.weights.policy_heads.policy_head_map[0]._proto.key
            == "test_policy"
        )
        policy_value = reloaded.weights.policy_heads.policy_head_map[
            0
        ].value.ip_pol_w.value
        assert np.isclose(policy_value[0], 15.0, rtol=1e-4)
    finally:
        os.unlink(tmp_path)
