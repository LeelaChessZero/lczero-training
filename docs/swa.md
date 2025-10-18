# Stochastic Weight Averaging (SWA)

Implementation plan for the Stochastic Weight Averaging (SWA).

## Configuration

In proto/training_config.proto, in `TrainingConfig`, add the field:

```proto
    // Stochastic Weight Averaging (SWA) configuration.
    optional SWAConfig = XX;

    message SWAConfig {
        uint32 period_steps = 1; // Number of steps between SWA updates.
        uint32 num_averages = 2; // Number of model snapshots to average.
    }
```

In `proto/export_config.proto`, in `ExportConfig`, add the field:

```proto
   bool export_swa_model = XX; // Whether to export the SWA model.
```

## `JitTrainingState` changes

We are adding new items:

* `swa_state: Optional[nnx.State]` to the `JitTrainingState` dataclass in
  `src/lczero_training/training/state.py`. This will have the same structure as
  the main model weights (or None).
* `num_averages: float` to track how many model snapshots have been averaged.

`TrainingState.new_from_config` will initialize these appropriately.

## Description

The SWA model does not affect the "main" training loop. It's a separate set of
weights that are updated in parallel. However, it gets exported if
`export_swa_model` is true.

In `Training.run` in `src/lczero_training/training/training.py`, we update the swa
model in the following cases:

* After every `swa_config.period_steps` steps (i.e. if the step count starts at
  0, then when `(step + 1) % period_steps == 0`)
* After training the epoch (except if already done above)

It goes as follows  (pseudocode):

```python
# weight may be fractional if we are doing the final update at the end of
# training and didn't have a full period since the last update.

last_model_weight: float = time_since_last_swa_update / period_steps

new_swa_weight = (
    num_averages / (num_averages + last_model_weight) * swa_state +
    last_model_weight / (num_averages + last_model_weight) * current_model_state)
num_averages = min(config.swa_config.num_averages, num_averages + last_model_weight)
```

The swa update must be a separate function, and not just code in run().

When exporting the model in `src/lczero_training/training/export.py`, if
`export_swa_model` is true, we export the SWA model instead of the main
model.
