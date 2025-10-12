# Checkpoint Migration

When part of the model or training setup changes, JAX training state checkpoints
may become incompatible with the new setup.

For this, we provide a utility to help migrate checkpoints to the new setup.

It's located in `src/lczero_training/training/migrate_checkpoint.py`, and is
called from `src/lczero_training/training/__main__.py`.

## Command line arguments

* `--config`: Path to the RootConfig textproto config. `model`/`training`
  sections of this config will be used to initialize the new training state.
  `checkpoint` is the location of the old checkpoint to migrate.
* `--new_checkpoint`: Path to save the new checkpoint to. If not set, the tool
  only checks whether the migration rules fully cover the differences between
  the old and new training states.
* `--overwrite`: If set, allows overwriting existing checkpoint. Also in this
  case, if `--new_checkpoint` is not set, old checkpoint is used as the new
  checkpoint.
* `--rules_file`: Path to a CheckpointMigrationConfig textproto file containing
  the migration rules. See below for the format of this file. If not set, no
  migration rules will be applied (used for debugging, or to check that the
  old and new states are identical).
* `--no-serialized_model`: By default, use serialized state for a model.
  Checkpoint already loads serialized as we don't provide schema. This is needed
  to avoid `GetAttrKey`s.
* `--checkpoint_step`: If set, use this step when loading from old checkpoint
  instead of the latest.
* `--new_checkpoint_step`: If set, use this step when saving the new checkpoint
  instead of copying the old step.

## Creation of the state to compare

The checkpoint is loaded into a raw pytree, i.e. template is not passed to it.
(old checkpoint with new model would fail to load).

Roughly, like this:

```python
    manager = ocp.CheckpointManager(
        filepath,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    state = manager.restore(step=None)
```

The model state is created using `TrainingState.new_from_config`
(`src/lczero_training/training/state.py`). Unless `--no-serialized_model` is
passed, the model state is serialized using `flax.serialization.to_state_dict`.

## Migration rules format

The migration rules are specified in a `CheckpointMigrationConfig` textproto
file. `CheckpointMigrationConfig` is defined in
`proto/checkpoint_migration_config.proto`

It contains a list of `CheckpointMigrationRule` messages.

Every rules has a `from_path` and `to_path` field (both optional, but at least
one must be set). These are string fields, which are json list of strings and
integers, and are mapped to pytree KeyPaths:

* Integers are used as `SequenceKey` (list/tuple indices).
* Strings are used as `DictKey` (dict keys).
* If other types is met in the path, an error is raised.
* If other key types are met in `PathKey`, the error is also raised.

* By default, all keys that are present both in the old and new state are
  preserved (copied from old to new).
* If both `from_path` and `to_path` are set, they must be different. The old
  values at `from_path` are copied to `to_path` in the new state.
* If only `to_path` is set, it means that the new state at `to_path` is
  taken from the new initialized state.
* If only `from_path` is set, it means that the old state at `from_path` is not
  present in the new state, and is ignored. Without this rule, the migration
  would fail if the old state has keys that are not present in the new state.
* If we want to keep a initialized ("new") value at a path that is also
  present in the old state, we use two rules: one with only `to_path` to
  initialize it, and one with only `from_path` to ignore the old value at
  that path.

Note that the `from_path` and `to_path` are prefixes of the actual paths, so the
actual subtrees are copied/ignored.

## Working of the migration

1. Both new and old states are created.
2. Both of them are flattened using `jax.tree_util.tree_flatten_with_path` into
   list of `(KeyPath, value)` pairs and a treedef.
3. Lists of `(KeyPath, value)` are converted to dicts mapping `KeyPath` to
   `value`.
4. We build a set of `source_paths` (keys of the old state).
5. We build a set of `dest_paths` (keys of the new state).
6. Rules are applied in arbitrary order as following:
   * If both `from_path` and `to_path` are set, the values prefixed by
     `from_path` are copied to corresponding `to_path` in the new state. Of
     course, `to_path` in the new state must exist (i.e. we never create new
     keys). The copied source paths are removed from `source_paths`.
     The corresponding destination paths are removed from `dest_paths`.
   * If only `to_path` is set, we delete all paths prefixed by `to_path` from
     `dest_paths`. This means that we keep the initialized value at this path.
   * If only `from_path` is set, we delete all paths prefixed by `from_path`
     from `source_paths`. This means that we ignore the old value at this path.
7. After all rules are applied, we check that `source_paths` and `dest_paths`
   are equal. If not, the migration is incomplete, and we raise an error.
8. After this, we copy all remaining `source_paths` (i.e. those that were not
   mentioned in any rule) to the new state. This means that by default, all
   keys that are present both in the old and new state are preserved (copied
   from old to new).
9. Finally, we unflatten the new state dict back to a pytree using the new
   treedef.

If `--new_checkpoint` is set, we save the new state to the specified path.
Otherwise, we just print that the migration is possible with the given rules.

Instead of just printing one error, the tool should print ALL errors it finds.
If should be helpful to fix the rules.

## Implementation notes

* There is `justfile`, e.g. `just build-proto` to build the protos.
* We use `uv`. E.g. after it's implemented, test with:

```bash
uv run python -m lczero_training.training.migrate_checkpoint \
    --config=~/tmp/lc0/config/overfit.textproto
```
