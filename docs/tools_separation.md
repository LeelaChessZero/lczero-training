# Tools Separation Plan

## CLI Layout (post-extraction)
- Commands are exposed via console scripts configured in `[project.scripts]`.
- Use `uv run <command> …` to execute tools without relying on `python -m`.
- Legacy `__main__.py` dispatchers have been removed.

## Commands To Extract

| CLI scope | Current entry | Underlying logic | Proposed new module | Style | Notes |
| --- | --- | --- | --- | --- | --- |
| Convert weights | `python -m lczero_training convert leela2jax …` | `convert/leela_to_jax.py::leela_to_jax_files` | `commands/leela2jax.py` | Wrapper | Only one subcommand; keep logic in `convert/` so tests remain unaffected. |
| Training init | `python -m lczero_training training init …` | `training/init.py::init` | `commands/training_init.py` | Wrapper | Heavy configuration/IO stays under `training/`. |
| Training run | `python -m lczero_training training train …` | `training/training.py::train` | `commands/training_run.py` | Wrapper | Start/stop logic is central, migrate later in plan. |
| Eval | `python -m lczero_training training eval …` | `training/eval.py::eval` | `commands/training_eval.py` | Wrapper | CLI arguments already map 1:1 to function inputs. |
| Tune LR | `python -m lczero_training training tune_lr …` | `training/tune_lr.py::tune_lr` | Wrapper | Produces CSV/plots; keep computational bits reusable. |
| Overfit probe | `python -m lczero_training training overfit …` | `training/overfit.py::overfit` | Wrapper | Mainly orchestration logic; wrapper keeps API stable. |
| Describe model | `python -m lczero_training training describe …` | `training/describe.py::describe` | Wrapper | Simple read-only command, safe early migration. |
| Data loader probe | `python -m lczero_training training test-dataloader …` | `training/dataloader_probe.py::probe_dataloader` | Wrapper | Good candidate for early extraction; minimal dependencies. |
| Checkpoint migrate | `python -m lczero_training training migrate-checkpoint …` | `training/migrate_checkpoint.py::migrate_checkpoint` | Wrapper | Needs careful option wiring; schedule after easier tools. |
| Daemon | `python -m lczero_training daemon` | `daemon/daemon.py::TrainingDaemon` | Single-file | Entry point only instantiates and runs the daemon; better to host the bootstrap in `commands/daemon.py` with logging config alongside. |
| TUI | `python -m lczero_training tui …` | `tui/app.py::TrainingTuiApp` | Single-file | The CLI should live in `commands/tui.py`, keeping awareness that it launches/controls the daemon over IPC. |

The new `src/lczero_training/commands/` package will hold these entrypoints.
Wrapper-style files only translate CLI flags into calls into the existing
implementation modules. Single-file entries (daemon, tui) can own their small
amount of glue code without introducing extra indirection.

## `pyproject.toml` / `uv` Integration
- Add a `[project.scripts]` table mapping short command names (e.g.
  `lczero-convert-leela2jax`, `lczero-train`, `lczero-train-init`) to the new
  functions in `lczero_training.commands`.
- `uv run <script-name> …` should remain the documented path, but once the
  scripts exist they also work when the package is installed or a venv is
  activated manually (the console entry point resolves the interpreter for the
  active environment).
- Consider grouping related training commands under a consistent prefix (for
  example, `lczero-train-init`, `lczero-train-run`) so tab completion remains
  obvious.
- Remove reliance on `python -m …` in documentation once the commands work
  through the entry points, and add `just` recipes if we want shorthands that
  run `uv run` implicitly.

## Migration Phases
Each phase completes in its own branch/commit and must end with `just format`
and `just pre-commit`.

1. **Scaffold commands package**  
   Create `src/lczero_training/commands/`, add `__init__.py`, and factor out
   shared helpers (logging, argument utilities) used by multiple commands.

2. **Extract convert CLI**  
   Move the `leela2jax` CLI into `commands/leela2jax.py`, register the
   script, update docs, run `just format` and `just pre-commit`.

3. **Extract describe CLI**  
   Create `commands/describe-training.py` as a thin wrapper around
   `training.describe`. Update docs/tests and run the formatting/pre-commit duo.

4. **Extract data loader probe**  
   Create `commands/test-dataloader.py`, wire arguments, update
   scripts/doc references, and finish with `just format` + `just pre-commit`.

5. **Extract overfit CLI**  
   Add `commands/overfit.py`, ensure argument parity, update docs,
   and run the required tooling.

6. **Extract training init CLI**  
   Introduce `commands/training-init.py`, migrate CLI options, adjust docs, run
   `just format` and `just pre-commit`.

7. **Extract evaluation CLI**  
   Move the eval entrypoint into `commands/training-eval.py`, verify flag
   wiring, refresh docs, run the required tooling.

8. **Extract learning-rate tuner**  
   Create `commands/tune-lr.py`, update documentation, run
   `just format` and `just pre-commit`.

9. **Extract checkpoint migration**  
   Build `commands/migrate-checkpoint.py`, carefully mirror argument
   behaviour, update docs/tests, run the tooling.

10. **Extract training run (main loop)**  
    Add `commands/train.py`, ensure there is no behaviour change, update
    references, run the tooling.

11. **Extract daemon CLI**  
    Move the daemon bootstrap into `commands/daemon.py`, keep logging set-up,
    update the TUI to import the helper, run the tooling.

12. **Extract TUI CLI**  
    Relocate the TUI entrypoint to `commands/tui.py`, verify daemon launch
    behaviour, update docs, run the tooling.

13. **Cleanup legacy dispatch**  
    Remove now-unused `__main__.py` entrypoints, consolidate docs, expand the
    scripts table, run `just format`, `just pre-commit`, and perform a final
    smoke test (`uv run <script>` as needed).

This sequencing starts with non-central tools and advances toward core and
interdependent CLIs (daemon/TUI), reducing risk while keeping every commit
small and verifiable.
