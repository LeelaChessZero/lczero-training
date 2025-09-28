# Training Pipeline Differences

This document captures the gaps between the legacy TensorFlow pipeline (using
configs like `/home/crem/Downloads/BT4-init1.yaml`) and the new
JAX-based training stack (`docs/example.textproto`,
`src/lczero_training/training/training.py`, and the associated protos).
We will iterate on these items one by one.

## Training Configuration

- **Mixed precision controls** (`precision`, `loss_scale`): drive dtype selection
  and loss scaling in `tf/tfprocess.py:210-224`, but are not represented in
  `proto/training_config.proto` or exercised in the JAX path, which currently
  always runs in full precision.
- **Stochastic Weight Averaging** (`swa`, `swa_max_n`, `swa_steps`): power the
  SWA accumulator and export logic in `tf/tfprocess.py:322-1157` and have no
  counterpart in the proto or JAX training loop.
- **Batch-norm renormalization** (`renorm`, `renorm_max_r`, `renorm_max_d`,
  `renorm_momentum`): configure batch-norm behaviour at
  `tf/tfprocess.py:327-1184`, with no fields or implementation in the new
  pipeline.
- **Learning-rate & reporting schedule** (`lr_values`, `lr_boundaries`,
  `warmup_steps`, `total_steps`, `test_steps`, `validation_steps`,
  `checkpoint_steps`, `train_avg_report_steps`): orchestrate the legacy training
  loop (`tf/tfprocess.py:597-980`). The new stack only exposes a constant LR and
  fixed `steps_per_network`, lacking these scheduling hooks.
- **Additional loss controls** (`q_ratio`, fine-grained `loss_weights`,
  `reg`): feed the TensorFlow loss mixer (`tf/tfprocess.py:485-558`). The proto
  currently only covers the main policy/value/moves-left heads, so optimistic,
  opponent, next-policy, and regularization weights are absent.
- **Optimizer toggles** (`lookahead_optimizer`, `new_optimizer`): enable
  alternative optimizer wrappers at `tf/tfprocess.py:399-430`, but are not
  surfaced in the new optimizer factory.

## Model Configuration

- **Legacy head/input switches** (`policy`, `value`, `moves_left`, `input_type`,
  BT3 feature toggles): select among multiple architecture variants in
  `tf/tfprocess.py:225-260`. The new `ModelConfig` fixes the architecture to the
  transformer stack and exposes none of these options.
- **Dropout & virtual batches** (`dropout_rate`, `virtual_batch_size`): are
  honoured by TensorFlow at `tf/tfprocess.py:209-213` and within the attention
  layers, but are unused and unconfigurable in the new model implementation.
- **Arc encoding and input gating** (`arc_encoding`, `input_gate`): influence
  TensorFlow model construction (`tf/tfprocess.py:191-198`) and are absent from
  the proto-driven model builder.

## Gradient Clipping

- `max_grad_norm` actively clips gradients in the TensorFlow stack
  (`tf/tfprocess.py:806-808`) and is set to `10` in the BT4 config. The new JAX
  training step (`src/lczero_training/training/training.py:111-117`) applies
  optimizer updates without any clipping, so this functionality still needs to
  be replicated alongside the corresponding proto fields.
