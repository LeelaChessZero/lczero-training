"""Wrapper classes for pythonic access to Lc0 weight protobufs."""

import gzip
from typing import Any, Iterator

import numpy as np
from google.protobuf.message import Message

from proto import net_pb2

from . import weight_codecs


class LayerWrapper:
    """Wraps a net_pb2.Weights.Layer with lazy float32 decoding."""

    __slots__ = ("_proto", "_fallback_encoding", "_cached_array", "_modified")

    def __init__(
        self, proto: net_pb2.Weights.Layer, fallback_encoding: int
    ) -> None:
        object.__setattr__(self, "_proto", proto)
        object.__setattr__(self, "_fallback_encoding", fallback_encoding)
        object.__setattr__(self, "_cached_array", None)
        object.__setattr__(self, "_modified", False)

    _proto: net_pb2.Weights.Layer
    _fallback_encoding: int
    _cached_array: np.ndarray | None
    _modified: bool

    @property
    def value(self) -> np.ndarray:
        """Decode to float32 on first access."""
        if self._cached_array is None:
            decoded = weight_codecs.decode_layer(
                self._proto, self._fallback_encoding
            )
            object.__setattr__(self, "_cached_array", decoded)
        assert self._cached_array is not None
        return self._cached_array

    @value.setter
    def value(self, arr: np.ndarray) -> None:
        """Set new array value, mark as modified."""
        object.__setattr__(self, "_cached_array", arr.astype(np.float32))
        object.__setattr__(self, "_modified", True)

    def commit(self, encoding: int) -> None:
        """Re-encode array to proto if modified."""
        if self._modified and self._cached_array is not None:
            params, min_val, max_val = weight_codecs.encode_layer(
                self._cached_array, encoding
            )
            self._proto.params = params
            self._proto.min_val = min_val
            self._proto.max_val = max_val
            # Note: encoding field left unset - use global net.format.weights_encoding
            del self._proto.dims[:]
            self._proto.dims.extend(self._cached_array.shape)
            object.__setattr__(self, "_modified", False)

    def __add__(self, other: "LayerWrapper") -> "LayerWrapper":
        if not isinstance(other, LayerWrapper):
            raise TypeError("Can only add LayerWrapper to LayerWrapper")
        result = LayerWrapper(net_pb2.Weights.Layer(), self._fallback_encoding)
        result.value = self.value + other.value
        return result

    def __sub__(self, other: "LayerWrapper") -> "LayerWrapper":
        if not isinstance(other, LayerWrapper):
            raise TypeError("Can only subtract LayerWrapper from LayerWrapper")
        result = LayerWrapper(net_pb2.Weights.Layer(), self._fallback_encoding)
        result.value = self.value - other.value
        return result

    def __mul__(self, scalar: float) -> "LayerWrapper":
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply LayerWrapper by scalar")
        result = LayerWrapper(net_pb2.Weights.Layer(), self._fallback_encoding)
        result.value = self.value * scalar
        return result

    def __rmul__(self, scalar: float) -> "LayerWrapper":
        return self.__mul__(scalar)


class ListWrapper:
    """Wraps protobuf repeated fields."""

    __slots__ = ("_proto_list", "_parent", "_item_cache")

    def __init__(self, proto_list: Any, parent: "NetWrapper") -> None:
        self._proto_list = proto_list
        self._parent = parent
        self._item_cache: dict[int, Any] = {}

    _proto_list: Any
    _parent: "NetWrapper"
    _item_cache: dict[int, Any]

    def __len__(self) -> int:
        return len(self._proto_list)

    def __getitem__(self, idx: int) -> Any:
        if idx not in self._item_cache:
            item_proto = self._proto_list[idx]
            self._item_cache[idx] = self._parent._wrap_field(item_proto)
        return self._item_cache[idx]

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]


class NetWrapper:
    """Wraps net_pb2.Net or nested Message types."""

    __slots__ = ("_proto", "_fallback_encoding", "_attr_cache")

    def __init__(
        self, proto_msg: Message, fallback_encoding: int | None = None
    ) -> None:
        object.__setattr__(self, "_proto", proto_msg)
        object.__setattr__(
            self,
            "_fallback_encoding",
            fallback_encoding or self._detect_encoding(),
        )
        object.__setattr__(self, "_attr_cache", {})

    _proto: Message
    _fallback_encoding: int
    _attr_cache: dict[str, Any]

    def _detect_encoding(self) -> int:
        """Extract encoding from net.format.weights_encoding."""
        if hasattr(self._proto, "format") and self._proto.HasField("format"):
            return self._proto.format.weights_encoding
        return net_pb2.Weights.Layer.LINEAR16

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if name in self._attr_cache:
            return self._attr_cache[name]

        if not hasattr(self._proto, name):
            raise AttributeError(
                f"{type(self._proto).__name__} has no field '{name}'"
            )

        value = getattr(self._proto, name)
        wrapped = self._wrap_field(value)
        self._attr_cache[name] = wrapped
        return wrapped

    def _wrap_field(self, value: Any) -> Any:
        """Determine wrapper type based on proto field."""
        if isinstance(value, net_pb2.Weights.Layer):
            return LayerWrapper(value, self._fallback_encoding)
        elif isinstance(value, Message):
            return NetWrapper(value, self._fallback_encoding)
        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return ListWrapper(value, self)
        else:
            return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if isinstance(value, NetWrapper):
            getattr(self._proto, name).CopyFrom(value._proto)
            self._attr_cache[name] = value
        elif isinstance(value, LayerWrapper):
            getattr(self._proto, name).CopyFrom(value._proto)
            self._attr_cache[name] = value
        else:
            setattr(self._proto, name, value)

    def save(self, path: str, encoding: int | None = None) -> None:
        """Save to file, committing all modified layers."""
        if encoding is None:
            encoding = net_pb2.Weights.Layer.FLOAT16

        self._commit_all(encoding)

        serialized = self._proto.SerializeToString()
        if path.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(serialized)
        else:
            with open(path, "wb") as f:
                f.write(serialized)

    def _commit_all(self, encoding: int) -> None:
        """Recursively commit all modified LayerWrappers."""
        for cached_value in self._attr_cache.values():
            if isinstance(cached_value, LayerWrapper):
                cached_value.commit(encoding)
            elif isinstance(cached_value, NetWrapper):
                cached_value._commit_all(encoding)
            elif isinstance(cached_value, ListWrapper):
                for item in cached_value:
                    if isinstance(item, NetWrapper):
                        item._commit_all(encoding)
                    elif isinstance(item, LayerWrapper):
                        item.commit(encoding)

    def __add__(self, other: "NetWrapper") -> "NetWrapper":
        """Element-wise addition of two networks."""
        if not isinstance(other, NetWrapper):
            raise TypeError("Can only add NetWrapper to NetWrapper")

        result_proto = type(self._proto)()
        result_proto.CopyFrom(self._proto)
        result = NetWrapper(result_proto, self._fallback_encoding)
        result._add_weights(self, other)
        return result

    def __sub__(self, other: "NetWrapper") -> "NetWrapper":
        """Element-wise subtraction of two networks."""
        if not isinstance(other, NetWrapper):
            raise TypeError("Can only subtract NetWrapper from NetWrapper")

        result_proto = type(self._proto)()
        result_proto.CopyFrom(self._proto)
        result = NetWrapper(result_proto, self._fallback_encoding)
        result._sub_weights(self, other)
        return result

    def __mul__(self, scalar: float) -> "NetWrapper":
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply NetWrapper by scalar")

        result_proto = type(self._proto)()
        result_proto.CopyFrom(self._proto)
        result = NetWrapper(result_proto, self._fallback_encoding)
        result._mul_weights(self, scalar)
        return result

    def __rmul__(self, scalar: float) -> "NetWrapper":
        return self.__mul__(scalar)

    def _add_weights(self, lhs: "NetWrapper", rhs: "NetWrapper") -> None:
        """Recursively add weights from lhs and rhs into self."""
        for field_desc in lhs._proto.DESCRIPTOR.fields:
            field_name = field_desc.name
            if not hasattr(lhs._proto, field_name):
                continue

            # Skip optional fields that are not set in both inputs.
            if not field_desc.is_required and not field_desc.is_repeated:
                lhs_has = lhs._proto.HasField(field_name)
                rhs_has = rhs._proto.HasField(field_name)
                if not lhs_has or not rhs_has:
                    continue

            lhs_val = getattr(lhs, field_name)
            rhs_val = getattr(rhs, field_name)

            if isinstance(lhs_val, LayerWrapper):
                self_layer = getattr(self, field_name)
                self_layer.value = lhs_val.value + rhs_val.value
            elif isinstance(lhs_val, NetWrapper):
                self_wrapper = getattr(self, field_name)
                self_wrapper._add_weights(lhs_val, rhs_val)
            elif isinstance(lhs_val, ListWrapper):
                self_list = getattr(self, field_name)
                # Only process indices that exist in both lists.
                min_len = min(len(lhs_val), len(rhs_val))
                for i in range(min_len):
                    if isinstance(lhs_val[i], (NetWrapper, LayerWrapper)):
                        if isinstance(lhs_val[i], LayerWrapper):
                            self_list[i].value = (
                                lhs_val[i].value + rhs_val[i].value
                            )
                        else:
                            self_list[i]._add_weights(lhs_val[i], rhs_val[i])
                # Truncate to min_len to avoid incomplete entries.
                del self_list._proto_list[min_len:]

    def _sub_weights(self, lhs: "NetWrapper", rhs: "NetWrapper") -> None:
        """Recursively subtract weights rhs from lhs into self."""
        for field_desc in lhs._proto.DESCRIPTOR.fields:
            field_name = field_desc.name
            if not hasattr(lhs._proto, field_name):
                continue

            # Skip optional fields that are not set in both inputs.
            if not field_desc.is_required and not field_desc.is_repeated:
                lhs_has = lhs._proto.HasField(field_name)
                rhs_has = rhs._proto.HasField(field_name)
                if not lhs_has or not rhs_has:
                    continue

            lhs_val = getattr(lhs, field_name)
            rhs_val = getattr(rhs, field_name)

            if isinstance(lhs_val, LayerWrapper):
                self_layer = getattr(self, field_name)
                self_layer.value = lhs_val.value - rhs_val.value
            elif isinstance(lhs_val, NetWrapper):
                self_wrapper = getattr(self, field_name)
                self_wrapper._sub_weights(lhs_val, rhs_val)
            elif isinstance(lhs_val, ListWrapper):
                self_list = getattr(self, field_name)
                # Only process indices that exist in both lists.
                min_len = min(len(lhs_val), len(rhs_val))
                for i in range(min_len):
                    if isinstance(lhs_val[i], (NetWrapper, LayerWrapper)):
                        if isinstance(lhs_val[i], LayerWrapper):
                            self_list[i].value = (
                                lhs_val[i].value - rhs_val[i].value
                            )
                        else:
                            self_list[i]._sub_weights(lhs_val[i], rhs_val[i])
                # Truncate to min_len to avoid incomplete entries.
                del self_list._proto_list[min_len:]

    def _mul_weights(self, source: "NetWrapper", scalar: float) -> None:
        """Recursively multiply all weights by scalar."""
        for field_desc in source._proto.DESCRIPTOR.fields:
            field_name = field_desc.name
            if not hasattr(source._proto, field_name):
                continue

            # Skip unset optional fields to avoid creating them in output.
            if (
                not field_desc.is_required
                and not field_desc.is_repeated
                and not source._proto.HasField(field_name)
            ):
                continue

            src_val = getattr(source, field_name)

            if isinstance(src_val, LayerWrapper):
                self_layer = getattr(self, field_name)
                self_layer.value = src_val.value * scalar
            elif isinstance(src_val, NetWrapper):
                self_wrapper = getattr(self, field_name)
                self_wrapper._mul_weights(src_val, scalar)
            elif isinstance(src_val, ListWrapper):
                self_list = getattr(self, field_name)
                for i in range(len(src_val)):
                    if isinstance(src_val[i], (NetWrapper, LayerWrapper)):
                        if isinstance(src_val[i], LayerWrapper):
                            self_list[i].value = src_val[i].value * scalar
                        else:
                            self_list[i]._mul_weights(src_val[i], scalar)
