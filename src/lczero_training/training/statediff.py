"""Utility to diff training state checkpoints."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterator, Tuple, Union

import numpy as np
import orbax.checkpoint as ocp
from google.protobuf import text_format

from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig

ShapeDtypeStruct: type[Any] | None = None
JAX_ARRAY_TYPES: tuple[type[Any], ...] = ()
try:  # pragma: no cover - optional dependency typing support
    import jax  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - jax missing in some envs
    pass
else:
    ShapeDtypeStruct = getattr(jax, "ShapeDtypeStruct", None)
    array_types: list[type[Any]] = []
    try:
        from jax import Array as _JaxArray  # type: ignore[attr-defined]
    except Exception:
        pass
    else:
        array_types.append(_JaxArray)  # type: ignore[arg-type]
    try:
        import jaxlib  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        xla_extension_module = getattr(jaxlib, "xla_extension", None)
        array_impl = None
        if xla_extension_module is not None:
            array_impl = getattr(xla_extension_module, "ArrayImpl", None)
        if isinstance(array_impl, type):
            array_types.append(array_impl)  # type: ignore[arg-type]
    if array_types:
        JAX_ARRAY_TYPES = tuple(array_types)


logger = logging.getLogger(__name__)

ChildKey = Union[str, int]

_IGNORED_METADATA_KEYS = {
    "metadata",
    "item_metadata",
    "state_dict_metadata",
    "value_metadata",
}


def state_diff(
    config_filename: str,
    old: str = "checkpoint",
    new: str = "model",
    show_values: bool = False,
    full_missing_subtrees: bool = True,
) -> None:
    """Compare two training states and print the structural differences."""
    config = _load_config(config_filename)

    try:
        old_label, old_state = _resolve_target(old, config)
        new_label, new_state = _resolve_target(new, config)
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    print(f"Comparing {old_label} -> {new_label}")
    diff_lines = list(
        _diff_structures(
            old_state,
            new_state,
            show_values=show_values,
            full_missing_subtrees=full_missing_subtrees,
        )
    )
    if not diff_lines:
        print("No differences found.")
    else:
        for line in diff_lines:
            print(line)


def _load_config(config_filename: str) -> RootConfig:
    config = RootConfig()
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)
    return config


def _resolve_target(name: str, config: RootConfig) -> tuple[str, Any]:
    name = name.strip()
    if name == "model":
        logger.info("Creating template training state from configuration")
        state = TrainingState.new_from_config(
            model_config=config.model, training_config=config.training
        )
        return ("model (empty state from config)", state)
    if name == "checkpoint":
        checkpoint_path = config.training.checkpoint.path
        if not checkpoint_path:
            raise ValueError(
                "Checkpoint path must be set in configuration to use 'checkpoint'."
            )
        logger.info("Restoring checkpoint from config path %s", checkpoint_path)
        state = _restore_checkpoint(checkpoint_path)
        return (f"checkpoint (latest from {checkpoint_path})", state)

    logger.info("Restoring checkpoint from path %s", name)
    state = _restore_checkpoint(name)
    return (f"checkpoint ({name})", state)


def _restore_checkpoint(path: str) -> Any:
    if not path:
        raise ValueError("Checkpoint path cannot be empty.")
    manager = ocp.CheckpointManager(
        path,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    state = manager.restore(step=None)
    if state is None:
        raise ValueError(f"No checkpoint available at {path}.")
    return state


def _diff_structures(
    old: Any,
    new: Any,
    *,
    show_values: bool,
    full_missing_subtrees: bool,
) -> Iterator[str]:
    yield from _diff_recursive(
        old,
        new,
        path=(),
        show_values=show_values,
        full_missing_subtrees=full_missing_subtrees,
    )


def _diff_recursive(
    old: Any,
    new: Any,
    *,
    path: Tuple[ChildKey, ...],
    show_values: bool,
    full_missing_subtrees: bool,
) -> Iterator[str]:
    old = _unwrap_value_wrapper(old)
    new = _unwrap_value_wrapper(new)

    if _structures_equal(old, new):
        return

    old_children = _get_children(old)
    new_children = _get_children(new)

    if old_children is not None and new_children is not None:
        keys = set(old_children).union(new_children)
        for key in sorted(keys, key=_sort_child_key):
            if key not in old_children:
                yield from _report_only_in_new(
                    new_children[key],
                    path + (key,),
                    show_values=show_values,
                    full_missing_subtrees=full_missing_subtrees,
                )
            elif key not in new_children:
                yield from _report_only_in_old(
                    old_children[key],
                    path + (key,),
                    show_values=show_values,
                    full_missing_subtrees=full_missing_subtrees,
                )
            else:
                yield from _diff_recursive(
                    old_children[key],
                    new_children[key],
                    path=path + (key,),
                    show_values=show_values,
                    full_missing_subtrees=full_missing_subtrees,
                )
        return

    if old_children is not None and new_children is None:
        location = _format_path(path)
        yield (
            f"- Structure mismatch at {location}: container {_describe_container(old)}"
            f" vs {_describe_leaf(new, show_values)}"
        )
        yield from _report_only_in_old(
            old,
            path,
            show_values=show_values,
            full_missing_subtrees=full_missing_subtrees,
        )
        return

    if old_children is None and new_children is not None:
        location = _format_path(path)
        yield (
            f"- Structure mismatch at {location}: {_describe_leaf(old, show_values)}"
            f" vs container {_describe_container(new)}"
        )
        yield from _report_only_in_new(
            new,
            path,
            show_values=show_values,
            full_missing_subtrees=full_missing_subtrees,
        )
        return

    # Both leaves
    if not _leaves_equal(old, new):
        location = _format_path(path)
        yield (
            f"- {location}: {_describe_leaf_difference(old, new, show_values)}"
        )


def _structures_equal(old: Any, new: Any) -> bool:
    if old is new:
        return True

    old = _unwrap_value_wrapper(old)
    new = _unwrap_value_wrapper(new)

    old_children = _get_children(old)
    new_children = _get_children(new)
    if old_children is not None or new_children is not None:
        if old_children is None or new_children is None:
            return False
        if set(old_children.keys()) != set(new_children.keys()):
            return False
        for key in old_children:
            if not _structures_equal(old_children[key], new_children[key]):
                return False
        return True

    if _is_array_like(old) and _is_array_like(new):
        return _array_metadata(old) == _array_metadata(new) and _arrays_equal(
            old, new
        )
    if type(old) is not type(new):
        return False
    if _is_array_like(old) or _is_array_like(new):
        return False
    return _safe_equals(old, new)


def _get_children(value: Any) -> Dict[ChildKey, Any] | None:
    value = _unwrap_value_wrapper(value)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _unwrap_value_wrapper(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        normalized: Dict[ChildKey, Any] = {}
        for key, child in value.items():
            if _is_metadata_key(key):
                continue
            normalized[_normalize_child_key(key)] = _unwrap_value_wrapper(child)
        return normalized
    if _is_array_like(value):
        return None
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return {
            index: _unwrap_value_wrapper(element)
            for index, element in enumerate(value)
        }
    return None


def _sort_child_key(key: ChildKey) -> tuple[int, Union[int, str]]:
    if isinstance(key, int):
        return (0, key)
    return (1, key)


def _normalize_child_key(key: ChildKey) -> ChildKey:
    if isinstance(key, str) and key.isdigit():
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _is_metadata_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key in _IGNORED_METADATA_KEYS or key.startswith("_")
    )


def _unwrap_value_wrapper(value: Any) -> Any:
    current = value
    while isinstance(current, Mapping):
        meaningful_keys = [
            key for key in current.keys() if not _is_metadata_key(key)
        ]
        if len(meaningful_keys) == 1 and meaningful_keys[0] == "value":
            current = current["value"]
            continue
        break
    return current


def _report_only_in_old(
    value: Any,
    path: Tuple[ChildKey, ...],
    *,
    show_values: bool,
    full_missing_subtrees: bool,
) -> Iterator[str]:
    location = _format_path(path)
    yield f"- Only in old: {location}"
    if full_missing_subtrees:
        for leaf_path, leaf_value in _iter_leaves(value, path):
            yield (
                "    "
                + _format_path(leaf_path)
                + " = "
                + _describe_leaf(leaf_value, show_values)
            )
    else:
        yield "    " + _describe_container(_unwrap_value_wrapper(value))


def _report_only_in_new(
    value: Any,
    path: Tuple[ChildKey, ...],
    *,
    show_values: bool,
    full_missing_subtrees: bool,
) -> Iterator[str]:
    location = _format_path(path)
    yield f"- Only in new: {location}"
    if full_missing_subtrees:
        for leaf_path, leaf_value in _iter_leaves(value, path):
            yield (
                "    "
                + _format_path(leaf_path)
                + " = "
                + _describe_leaf(leaf_value, show_values)
            )
    else:
        yield "    " + _describe_container(_unwrap_value_wrapper(value))


def _iter_leaves(
    value: Any, path: Tuple[ChildKey, ...]
) -> Iterator[tuple[Tuple[ChildKey, ...], Any]]:
    value = _unwrap_value_wrapper(value)
    children = _get_children(value)
    if children is None:
        yield path, value
        return
    for key in sorted(children.keys(), key=_sort_child_key):
        yield from _iter_leaves(children[key], path + (key,))


def _describe_leaf(value: Any, show_values: bool) -> str:
    value = _unwrap_value_wrapper(value)
    if _is_array_like(value):
        shape, dtype = _array_metadata(value)
        if show_values:
            array_value = _maybe_to_numpy(value)
            if array_value is not None:
                value_str = np.array2string(
                    array_value, threshold=10, edgeitems=3
                )
            else:
                value_str = "<unavailable>"
            return f"Array(shape={shape}, dtype={dtype}, value={value_str})"
        return f"Array(shape={shape}, dtype={dtype})"
    if isinstance(value, (str, bytes, bytearray)):
        return repr(value) if show_values else f"{type(value).__name__}"
    if show_values:
        return repr(value)
    return type(value).__name__


def _describe_container(value: Any) -> str:
    value = _unwrap_value_wrapper(value)
    if dataclasses.is_dataclass(value):
        return f"{type(value).__name__} dataclass"
    if isinstance(value, Mapping):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return f"{type(value).__name__}(len={len(value)})"
    if _is_array_like(value):
        shape, dtype = _array_metadata(value)
        return f"Array(shape={shape}, dtype={dtype})"
    return type(value).__name__


def _describe_leaf_difference(old: Any, new: Any, show_values: bool) -> str:
    if _is_array_like(old) and _is_array_like(new):
        old_shape, old_dtype = _array_metadata(old)
        new_shape, new_dtype = _array_metadata(new)
        if (old_shape, old_dtype) != (new_shape, new_dtype):
            return (
                "array shape/dtype mismatch: "
                f"old shape={old_shape}, dtype={old_dtype}; "
                f"new shape={new_shape}, dtype={new_dtype}"
            )
        if show_values:
            old_val = _maybe_to_numpy(old)
            new_val = _maybe_to_numpy(new)
            old_repr = (
                np.array2string(old_val, threshold=10, edgeitems=3)
                if old_val is not None
                else "<unavailable>"
            )
            new_repr = (
                np.array2string(new_val, threshold=10, edgeitems=3)
                if new_val is not None
                else "<unavailable>"
            )
            return f"array values differ: old={old_repr}, new={new_repr}"
        return f"array values differ (shape={old_shape}, dtype={old_dtype})"

    if type(old) is not type(new):
        return f"type mismatch: {type(old).__name__} vs {type(new).__name__}"

    if show_values:
        return f"value mismatch: old={old!r}, new={new!r}"
    return f"value mismatch (type {type(old).__name__})"


def _array_metadata(value: Any) -> tuple[Any, Any]:
    if ShapeDtypeStruct is not None and isinstance(value, ShapeDtypeStruct):
        return value.shape, value.dtype
    array = _maybe_to_numpy(value)
    if array is not None:
        return array.shape, array.dtype
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return shape, dtype


def _maybe_to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if JAX_ARRAY_TYPES and isinstance(value, JAX_ARRAY_TYPES):
        return np.asarray(value)
    if hasattr(value, "__array__"):
        try:
            return np.asarray(value)
        except Exception:  # pragma: no cover - fallback
            return None
    return None


def _is_array_like(value: Any) -> bool:
    if ShapeDtypeStruct is not None and isinstance(value, ShapeDtypeStruct):
        return True
    if isinstance(value, np.ndarray):
        return True
    if JAX_ARRAY_TYPES and isinstance(value, JAX_ARRAY_TYPES):
        return True
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "__array__")
    ):
        return True
    return False


def _arrays_equal(lhs: Any, rhs: Any) -> bool:
    lhs_array = _maybe_to_numpy(lhs)
    rhs_array = _maybe_to_numpy(rhs)
    if lhs_array is None or rhs_array is None:
        return False
    return np.array_equal(lhs_array, rhs_array)


def _leaves_equal(old: Any, new: Any) -> bool:
    if _is_array_like(old) and _is_array_like(new):
        if _array_metadata(old) != _array_metadata(new):
            return False
        return _arrays_equal(old, new)
    if _is_array_like(old) != _is_array_like(new):
        return False
    if type(old) is not type(new):
        return False
    return _safe_equals(old, new)


def _safe_equals(lhs: Any, rhs: Any) -> bool:
    try:
        return bool(lhs == rhs)
    except Exception:  # pragma: no cover - safe comparison fallback
        return False


def _format_path(path: Tuple[ChildKey, ...]) -> str:
    if not path:
        return "<root>"
    parts: list[str] = []
    for part in path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            if parts:
                parts.append(".")
            parts.append(str(part))
    return "".join(parts)
