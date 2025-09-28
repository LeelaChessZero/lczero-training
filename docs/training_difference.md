# Training Pipeline Differences

## Context Summary
- Legacy training used TensorFlow (`tf/tfprocess.py`) with configs such as `/home/crem/Downloads/BT4-init1.yaml`; the rewrite uses JAX with proto-driven configs (`docs/example.textproto`, `src/lczero_training/training/training.py`).
- Some behaviours exist only on the historical `daniel/tf-214` branch (requires `git fetch daniel tf-214` + checkout). Those features are marked below so it’s clear when the current repository state differs from that branch.

## Implementation Phases
1. **Phase 1 – Mixed Precision Controls**
   - Re-introduce config for `precision` and `loss_scale` (`tf/tfprocess.py:210` and `tf/tfprocess.py:222`) so the JAX path can toggle compute dtype and scaling similar to TensorFlow.
   - Proto impact: extend `proto/training_config.proto` and update `docs/example.textproto` plus runtime handling in `src/lczero_training/training/training.py`.

2. **Phase 2 – Stochastic Weight Averaging (SWA)**
   - Mirror SWA tracking, checkpointing, and export (`tf/tfprocess.py:322-1157`, `daniel/tf-214`) in the JAX trainer, adding config fields for `swa`, `swa_max_n`, and `swa_steps`.

3. **Phase 3 – Batch-Norm Renormalization**
   - Support `renorm`, `renorm_max_r`, `renorm_max_d`, `renorm_momentum` as used when constructing batch norm layers (`tf/tfprocess.py:327-1184`, `daniel/tf-214`).

4. **Phase 4 – Training Schedule & Reporting Hooks**
   - Port learning-rate scheduling (`lr_values`, `lr_boundaries`, `warmup_steps`) and reporting cadence (`total_steps`, `test_steps`, `validation_steps`, `checkpoint_steps`, `train_avg_report_steps`) managed in `tf/tfprocess.py:597-980` (`daniel/tf-214`).

5. **Phase 5 – Extended Loss Weighting**
   - Recreate `q_ratio`, fine-grained loss weights, and regularization terms wired through `tf/tfprocess.py:485-558` (`daniel/tf-214`), updating the loss builder beyond the current `policy/value/movesleft` trio in `src/lczero_training/model/loss_function.py:24-84`.

6. **Phase 6 – Optimizer Variants**
   - Add config toggles like `lookahead_optimizer` and `new_optimizer` (`tf/tfprocess.py:399-430`, `daniel/tf-214`) alongside the base JAX optimizer factory (`src/lczero_training/training/optimizer.py:5-27`).

7. **Phase 7 – Legacy Head & Input Options**
   - Restore optional heads and input formats (`tf/tfprocess.py:225-260`, `daniel/tf-214`) so proto configs can pick classical heads or BT3 extras currently missing from `proto/model_config.proto:9-32` and `src/lczero_training/model/model.py:14-64`.

8. **Phase 8 – Dropout and Virtual Batches**
   - Surface `dropout_rate` and `virtual_batch_size` (handled in `tf/tfprocess.py:209-213`, `daniel/tf-214`) in the proto model config and ensure JAX layers respect them.

9. **Phase 9 – Arc Encoding & Input Gating**
   - Implement `arc_encoding` and `input_gate` hooks from `tf/tfprocess.py:191-198` (`daniel/tf-214`) within the new model stack.

10. **Phase 10 – Gradient Clipping**
    - Re-enable `max_grad_norm` clipping (`tf/tfprocess.py:806-808`, `daniel/tf-214`) by extending proto config and wrapping the Optax update in `src/lczero_training/training/training.py:111-117` with a clipping transformation.

11. **Phase 11 – Policy Loss Objective (KL vs CE)**
    - Match the KL-style policy loss that subtracts target entropy on `daniel/tf-214` (`tf/tfprocess.py:508-525`) instead of the current raw cross-entropy in `src/lczero_training/model/loss_function.py:70-84`.
