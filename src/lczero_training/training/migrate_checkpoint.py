from typing import Any, Dict, List, Set, Tuple

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization
from google.protobuf import text_format
from orbax.checkpoint.utils import tuple_path_from_keypath

from lczero_training.training import state as state_lib
from proto import checkpoint_migration_config_pb2, root_config_pb2


def _str_to_key_path(path_str: str) -> tuple[str, ...]:
    return tuple(path_str.split("."))


def _load_new_state(
    root_config: root_config_pb2.RootConfig, serialized_model: bool
) -> Any:
    new_state = state_lib.TrainingState.new_from_config(
        root_config.model, root_config.training
    )
    if serialized_model:
        return serialization.to_state_dict(new_state)
    return new_state


def load_checkpoint(
    checkpoint_path: str, checkpoint_step: int | None = None
) -> Tuple[Any, int]:
    """Load a checkpoint from the given path.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        checkpoint_step: Step to load, or None to load the latest.

    Returns:
        Tuple of (checkpoint_state, checkpoint_step).
    """
    manager = ocp.CheckpointManager(
        checkpoint_path,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    if checkpoint_step is None:
        checkpoint_step = manager.latest_step()
    if checkpoint_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")
    return manager.restore(checkpoint_step), checkpoint_step


def get_checkpoint_steps(
    checkpoint_path: str,
    min_step: int | None = None,
    max_step: int | None = None,
) -> list[int]:
    """Get all checkpoint steps in the given range.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        min_step: Minimum step (inclusive), or None for no minimum.
        max_step: Maximum step (inclusive), or None for no maximum.

    Returns:
        List of checkpoint steps in ascending order.
    """
    manager = ocp.CheckpointManager(
        checkpoint_path,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    all_steps = sorted(manager.all_steps())
    filtered_steps = []
    for step in all_steps:
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        filtered_steps.append(step)
    filtered_steps.sort()
    return filtered_steps


def _load_old_state(
    checkpoint_path: str, checkpoint_step: int | None
) -> Tuple[Any, int]:
    return load_checkpoint(checkpoint_path, checkpoint_step)


def load_migration_rules(rules_file: str | None) -> List[Tuple[Any, Any]]:
    """Load migration rules from a CheckpointMigrationConfig file.

    Args:
        rules_file: Path to the CheckpointMigrationConfig textproto file,
            or None to return empty rules.

    Returns:
        List of (from_path, to_path) tuples representing migration rules.
    """
    rules = []
    if rules_file:
        migration_config = (
            checkpoint_migration_config_pb2.CheckpointMigrationConfig()
        )
        with open(rules_file, "r") as f:
            text_format.Parse(f.read(), migration_config)
        for rule_proto in migration_config.rule:
            from_path = (
                _str_to_key_path(rule_proto.from_path)
                if rule_proto.from_path
                else None
            )
            to_path = (
                _str_to_key_path(rule_proto.to_path)
                if rule_proto.to_path
                else None
            )
            rules.append((from_path, to_path))
    return rules


def _format_value(value: Any) -> str:
    if isinstance(value, (np.ndarray, jax.Array)):
        return f"{value.dtype}{value.shape}"
    return repr(value)


def _format_path_diff(
    unhandled_source: Set[Tuple[str, ...]],
    unhandled_dest: Set[Tuple[str, ...]],
    old_paths: Dict[Tuple[str, ...], Any],
    new_paths: Dict[Tuple[str, ...], Any],
) -> str:
    diff = []
    for p in sorted(list(unhandled_source | unhandled_dest)):
        p_str = ".".join(p)
        if p in unhandled_source:
            diff.append(f"- {p_str}: {_format_value(old_paths[p])}")
        if p in unhandled_dest:
            diff.append(f"+ {p_str}: {_format_value(new_paths[p])}")
    return "\n".join(diff)


class Migration:
    def __init__(self, old_state: Any, new_state: Any):
        old_leaves = jax.tree_util.tree_leaves_with_path(old_state)
        new_flat, self.new_treedef = jax.tree_util.tree_flatten_with_path(
            new_state
        )

        self.old_paths: Dict[Tuple[str, ...], Any] = {
            tuple_path_from_keypath(path): value for path, value in old_leaves
        }
        self.new_paths: Dict[Tuple[str, ...], Any] = {
            tuple_path_from_keypath(path): value for path, value in new_flat
        }
        self.new_leaves: List[Any] = [value for _, value in new_flat]
        self.new_path_to_idx: Dict[Tuple[str, ...], int] = {
            tuple_path_from_keypath(path): i
            for i, (path, _) in enumerate(new_flat)
        }

        self.source_paths: Set[Tuple[str, ...]] = set(self.old_paths.keys())
        self.dest_paths: Set[Tuple[str, ...]] = set(self.new_path_to_idx.keys())

        print(f"{len(self.source_paths & self.dest_paths)} common keys")
        print(f"{len(self.source_paths - self.dest_paths)} keys disappeared")
        print(f"{len(self.dest_paths - self.source_paths)} keys appeared")

        self.errors: List[str] = []

    def _apply_move_rule(
        self, from_path: Tuple[str, ...], to_path: Tuple[str, ...]
    ) -> None:
        if from_path == to_path:
            self.errors.append(
                f"from_path and to_path are the same: {from_path}"
            )
            return

        source_prefixed = {
            p for p in self.source_paths if p[: len(from_path)] == from_path
        }
        dest_prefixed = {
            p for p in self.dest_paths if p[: len(to_path)] == to_path
        }

        if not source_prefixed:
            self.errors.append(f"from_path {from_path} not found in old state")
        if not dest_prefixed:
            self.errors.append(f"to_path {to_path} not found in new state")

        for p in source_prefixed:
            new_p = to_path + p[len(from_path) :]
            if new_p in self.dest_paths:
                idx = self.new_path_to_idx[new_p]
                self.new_leaves[idx] = self.old_paths[p]
                self.source_paths.remove(p)
                self.dest_paths.remove(new_p)
            else:
                self.errors.append(f"Path {new_p} not found in new state")

    def _apply_ignore_rule(self, from_path: Tuple[str, ...]) -> None:
        source_prefixed = {
            p for p in self.source_paths if p[: len(from_path)] == from_path
        }
        if not source_prefixed:
            self.errors.append(f"from_path {from_path} not found in old state")
        self.source_paths -= source_prefixed

    def _apply_keep_rule(self, to_path: Tuple[str, ...]) -> None:
        dest_prefixed = {
            p for p in self.dest_paths if p[: len(to_path)] == to_path
        }
        if not dest_prefixed:
            self.errors.append(f"to_path {to_path} not found in new state")
        self.dest_paths -= dest_prefixed

    def apply_rules(self, rules: List[Tuple[Any, Any]]) -> None:
        for from_path, to_path in rules:
            if from_path and to_path:
                self._apply_move_rule(from_path, to_path)
            elif from_path:
                self._apply_ignore_rule(from_path)
            elif to_path:
                self._apply_keep_rule(to_path)

    def run(self, rules: List[Tuple[Any, Any]]) -> Any:
        self.apply_rules(rules)

        # Copy remaining paths
        copied_paths = self.source_paths & self.dest_paths
        for p in copied_paths:
            idx = self.new_path_to_idx[p]
            self.new_leaves[idx] = self.old_paths[p]
        self.source_paths -= copied_paths
        self.dest_paths -= copied_paths

        unhandled_source = self.source_paths
        unhandled_dest = self.dest_paths

        if unhandled_source or unhandled_dest:
            self.errors.append(
                "Unmapped paths:\n"
                + _format_path_diff(
                    unhandled_source,
                    unhandled_dest,
                    self.old_paths,
                    self.new_paths,
                )
            )

        if self.errors:
            raise ValueError("\n".join(self.errors))

        return getattr(self.new_treedef, "unflatten")(self.new_leaves)


def _save_checkpoint(
    migrated_state: Any,
    new_checkpoint_path: str,
    new_checkpoint_step: int,
    overwrite: bool,
) -> None:
    manager = ocp.CheckpointManager(
        new_checkpoint_path,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            create=True, save_interval_steps=1, todelete_subdir="trash"
        ),
    )

    if new_checkpoint_step in manager.all_steps():
        if overwrite:
            manager.delete(new_checkpoint_step)
            manager.wait_until_finished()
        else:
            raise ValueError(
                f"Checkpoint already exists at {new_checkpoint_step} in "
                f"{new_checkpoint_path}. "
                "Use --overwrite to overwrite."
            )

    manager.save(new_checkpoint_step, migrated_state)
    manager.wait_until_finished()
    print(
        f"New checkpoint saved successfully to {new_checkpoint_path} at step "
        f"{new_checkpoint_step}."
    )


def migrate_checkpoint(
    config: str,
    new_checkpoint: str | None,
    overwrite: bool,
    rules_file: str | None,
    serialized_model: bool,
    checkpoint_step: int | None,
    new_checkpoint_step: int | None,
) -> None:
    """Migrates a checkpoint to a new training state."""
    root_config = root_config_pb2.RootConfig()
    with open(config, "r") as f:
        text_format.Parse(f.read(), root_config)

    new_state = _load_new_state(root_config, serialized_model)
    old_state, old_checkpoint_step = _load_old_state(
        root_config.training.checkpoint.path, checkpoint_step
    )
    rules = load_migration_rules(rules_file)

    migration = Migration(old_state, new_state)
    migrated_state = migration.run(rules)

    if new_checkpoint:
        checkpoint_path = new_checkpoint
    elif overwrite:
        checkpoint_path = root_config.training.checkpoint.path
    else:
        print("Migration check successful.")
        return

    if new_checkpoint_step is None:
        new_checkpoint_step = old_checkpoint_step

    _save_checkpoint(
        migrated_state, checkpoint_path, new_checkpoint_step, overwrite
    )
