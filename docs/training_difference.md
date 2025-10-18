# Training Pipeline Differences

## Context Summary
- Legacy training used TensorFlow (`tf/tfprocess.py`) driven by YAML configs such as `/home/crem/Downloads/BT4-init1.yaml`. The rewrite runs on JAX with proto configs (`docs/example.textproto`, `src/lczero_training/training/training.py`).
- Several behaviours only exist on the historical `daniel/tf-214` branch (add the remote with `git remote add daniel https://github.com/daniel-monroe/lczero-training.git` if it's missing, then fetch via `git fetch daniel tf-214` and checkout to inspect). These are flagged below.
- Goal: stabilise the current transformer configuration; active phases are ordered by how strongly they can destabilise training if left unimplemented.

## Active Phases for Current Config (ordered by suspected impact)
1. **Phase A – Gradient Clipping** *(present on `daniel/tf-214`)*
   - TensorFlow clips gradients using `max_grad_norm` before applying updates (`tf/tfprocess.py:806-808`). JAX applies raw Optax updates (`src/lczero_training/training/training.py:111-117`), so large batches are no longer bounded.

2. **Phase B – Training Schedule & Warmup** *(present on `daniel/tf-214`, now implemented)*
   - The JAX training pipeline now honours piecewise warmup schedules defined with `training.optimizer.lr_schedule`, matching the TensorFlow helper's multi-stage curves (`tf/tfprocess.py:597-980`). Pretrained runs can anchor the schedule by setting the first `starting_step` to the network's existing global step.

3. **Phase C – Policy Loss Objective (KL vs CE)** *(present on `daniel/tf-214`)*
   - The TensorFlow helper normalises targets, applies temperature, and subtracts target entropy, yielding `KL(target || policy)` (`tf/tfprocess.py:493-525`). The JAX loss keeps raw cross-entropy with masked negatives (`src/lczero_training/model/loss_function.py:70-84`), changing both gradient scale and objective.

4. **Phase D – Extended Loss Weighting & Regularisation** *(present on `daniel/tf-214`)*
   - Loss mixer accepts `q_ratio`, component weights, and `reg` term in `tf/tfprocess.py:485-558`. The JAX loss builder (`src/lczero_training/model/loss_function.py:24-84`) ignores these extras, optimising a different objective.

5. **Phase E – Batch-Norm Renormalisation** *(present on `daniel/tf-214`)*
   - Config options `renorm`, `renorm_max_r`, `renorm_max_d`, `renorm_momentum` feed batch norm setup (`tf/tfprocess.py:327-1184`). Without them, the JAX build always runs vanilla batch norm.

6. **Phase F – Dropout & Virtual Batch Size** *(present on `daniel/tf-214`)*
   - `dropout_rate` and `virtual_batch_size` influence attention blocks and batch-splitting checks (`tf/tfprocess.py:209-213`, `tf/tfprocess.py:728-733`). The JAX model does not read or honour these fields.

7. **Phase G – Diff Focus Sampling** *(present on `daniel/tf-214`)*
   - Diff focus parameters (`diff_focus_min`, `diff_focus_slope`, `diff_focus_q_weight`, `diff_focus_pol_scale`) prune training samples based on value/policy disagreement in `tf/chunkparser.py:400-468`. The new data loader lacks this sampler, so dataset composition shifts toward lower-disagreement positions.

8. **Phase H – Stochastic Weight Averaging (SWA)** *(present on `daniel/tf-214`)*
   - Legacy loop maintains SWA weights, checkpoints, and exports (`tf/tfprocess.py:322-1157`). JAX ignores `swa`/`swa_steps`/`swa_max_n`, preventing SWA networks during long runs.

9. **Phase I – Mixed Precision Controls**
   - TensorFlow supports `precision`/`loss_scale` toggles (`tf/tfprocess.py:210-224`). The JAX path always uses full precision; add proto fields and dtype handling if BF16/FP16 are still required.

## Inactive / Optional Phases (not exercised by current config)
- **Phase J – Optimizer Variants** *(requires `daniel/tf-214`)*: lookahead/new optimizer toggles (`tf/tfprocess.py:399-430`).
- **Phase K – Legacy Head & Input Options** *(requires `daniel/tf-214`)*: classical policy/value heads, BT3 extras (`tf/tfprocess.py:225-260`).
- **Phase L – Arc Encoding & Input Gating** *(requires `daniel/tf-214`)*: optional input preprocessing hooks (`tf/tfprocess.py:191-198`).
