"""Encoding and decoding logic for Lc0 weight formats."""

import numpy as np

from proto import net_pb2


def decode_linear16(
    params: bytes, min_val: float, max_val: float, shape: tuple[int, ...]
) -> np.ndarray:
    """Decode LINEAR16 format to float32 array."""
    raw = np.frombuffer(params, dtype=np.uint16)
    norm = raw.astype(np.float32) / 65535.0
    values = norm * max_val + (1.0 - norm) * min_val
    return values.reshape(shape[::-1]).transpose()


def encode_linear16(arr: np.ndarray) -> tuple[bytes, float, float]:
    """Encode float32 array to LINEAR16 format."""
    flat = arr.T.flatten().astype(np.float32)
    min_val = float(flat.min())
    max_val = float(flat.max())
    rng = max_val - min_val

    if rng < 1e-8:
        norm = np.full_like(flat, 0.5)
    else:
        norm = (flat - min_val) / rng

    quant = np.round(norm * 65535.0).astype(np.uint16)
    return quant.tobytes(), min_val, max_val


def decode_float16(params: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode FLOAT16 format to float32 array."""
    raw = np.frombuffer(params, dtype=np.float16)
    values = raw.astype(np.float32)
    return values.reshape(shape[::-1]).transpose()


def encode_float16(arr: np.ndarray) -> tuple[bytes, float, float]:
    """Encode float32 array to FLOAT16 format."""
    flat = arr.T.flatten().astype(np.float16)
    return flat.tobytes(), 0.0, 0.0


def decode_bfloat16(params: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode BFLOAT16 format to float32 array via bit manipulation."""
    raw_u16 = np.frombuffer(params, dtype=np.uint16)
    raw_u32 = raw_u16.astype(np.uint32) << 16
    values = raw_u32.view(np.float32)
    return values.reshape(shape[::-1]).transpose()


def encode_bfloat16(arr: np.ndarray) -> tuple[bytes, float, float]:
    """Encode float32 array to BFLOAT16 format via bit manipulation."""
    flat = arr.T.flatten().astype(np.float32)
    u32 = flat.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes(), 0.0, 0.0


def decode_layer(
    layer: net_pb2.Weights.Layer, fallback_encoding: int
) -> np.ndarray:
    """Decode a Layer protobuf to float32 NumPy array."""
    encoding = layer.encoding if layer.encoding else fallback_encoding

    if not layer.dims:
        size = len(layer.params) // 2
        shape: tuple[int, ...] = (size,)
    else:
        shape = tuple(layer.dims)

    if encoding == net_pb2.Weights.Layer.LINEAR16:
        return decode_linear16(
            layer.params, layer.min_val, layer.max_val, shape
        )
    elif encoding == net_pb2.Weights.Layer.FLOAT16:
        return decode_float16(layer.params, shape)
    elif encoding == net_pb2.Weights.Layer.BFLOAT16:
        return decode_bfloat16(layer.params, shape)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def encode_layer(arr: np.ndarray, encoding: int) -> tuple[bytes, float, float]:
    """Encode a float32 NumPy array to Layer format."""
    if encoding == net_pb2.Weights.Layer.LINEAR16:
        return encode_linear16(arr)
    elif encoding == net_pb2.Weights.Layer.FLOAT16:
        return encode_float16(arr)
    elif encoding == net_pb2.Weights.Layer.BFLOAT16:
        return encode_bfloat16(arr)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
