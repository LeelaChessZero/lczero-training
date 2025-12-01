from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optax

from proto.training_config_pb2 import LrSchedule


def _create_rule_fn(rule: LrSchedule) -> Callable:
    """
    Creates a JAX-compatible function for a single LR schedule rule.

    All data from the protobuf is extracted here and captured by the closure of
    the returned function. This avoids protobuf parsing inside the main schedule
    function which will be JIT-compiled.
    """
    start_step = float(rule.starting_step)
    durations = list(rule.duration_steps)
    lrs = list(rule.lr)
    is_looping = rule.loop

    # Handle simple cases where the LR is constant for this rule.
    if not durations or not lrs:
        lr_val = jnp.asarray(lrs[-1] if lrs else 0.0, dtype=jnp.float32)
        # Return a simple lambda that ignores the step and returns the constant value.
        return lambda step: lr_val

    period = sum(durations)
    if period == 0.0:
        lr_val = jnp.asarray(lrs[-1], dtype=jnp.float32)
        return lambda step: lr_val

    # Pre-calculate JAX arrays for use in the schedule function.
    transitions = [
        (
            rule.transition[i]
            if i < len(rule.transition)
            else LrSchedule.Transition.CONSTANT
        )
        for i in range(len(durations))
    ]

    durs_j = jnp.asarray(durations, dtype=jnp.float32)
    ends_j = jnp.cumsum(durs_j)
    starts_j = ends_j - durs_j

    # Interpolation start/end LRs for each segment.
    a_vals = [
        lrs[i] if i < len(lrs) else lrs[-1] for i in range(len(durations))
    ]
    b_vals = [
        lrs[i + 1] if (i + 1) < len(lrs) else a_vals[i]
        for i in range(len(durations))
    ]
    lrs_a_j = jnp.asarray(a_vals, dtype=jnp.float32)
    lrs_b_j = jnp.asarray(b_vals, dtype=jnp.float32)

    trans_j = jnp.asarray(transitions, dtype=jnp.int32)
    last_lr_j = jnp.asarray(lrs[-1], dtype=jnp.float32)
    period_j = jnp.asarray(period, dtype=jnp.float32)

    def rule_fn(step: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible function evaluating the LR for a given step."""
        rel_step = step - start_step

        if is_looping:
            rel_step = jnp.mod(rel_step, period_j)

        # Find active segment and calculate interpolation factor `t`.
        is_in_segment = (
            (rel_step >= starts_j) & (rel_step < ends_j) & (durs_j > 0)
        )
        # Use maximum() to avoid division by zero for zero-duration segments.
        t = jnp.clip((rel_step - starts_j) / jnp.maximum(durs_j, 1.0), 0.0, 1.0)

        # Calculate interpolated values for all segments for all transition types.
        lin = lrs_a_j + (lrs_b_j - lrs_a_j) * t
        cos = lrs_a_j + 0.5 * (1.0 - jnp.cos(jnp.pi * t)) * (lrs_b_j - lrs_a_j)

        # Select interpolation type for each segment. Default to CONSTANT (lrs_a_j).
        interp_vals = jnp.where(
            trans_j == LrSchedule.Transition.LINEAR, lin, lrs_a_j
        )
        interp_vals = jnp.where(
            trans_j == LrSchedule.Transition.COSINE, cos, interp_vals
        )

        # Select the value from the active segment by masking.
        lr = jnp.sum(interp_vals * is_in_segment)

        # If not in any segment (e.g., gap between segments), use the last LR value.
        lr = jnp.where(jnp.any(is_in_segment), lr, last_lr_j)

        if not is_looping:
            # For non-looping rules, if past the end, clamp to the last LR.
            lr = jnp.where(rel_step >= period_j, last_lr_j, lr)

        return lr

    return rule_fn


def make_lr_schedule(schedules: Sequence[LrSchedule]) -> optax.Schedule:
    """
    Creates a learning rate schedule from a sequence of LrSchedule protobufs.

    The schedule is composed of multiple rules, each active for a certain
    range of training steps.
    """
    if not schedules:
        return lambda count: jnp.asarray(0.0, dtype=jnp.float32)

    rule_fns = [_create_rule_fn(rule) for rule in schedules]
    start_steps = jnp.asarray(
        [rule.starting_step for rule in schedules], dtype=jnp.float32
    )

    # Determine the learning rate to use for steps before the first rule begins.
    first_lrs = jnp.asarray(
        [rule.lr[0] if rule.lr else 0.0 for rule in schedules],
        dtype=jnp.float32,
    )
    earliest_rule_idx = jnp.argmin(start_steps)
    pre_start_lr = first_lrs[earliest_rule_idx]
    min_start_step = start_steps[earliest_rule_idx]

    def schedule(count: jnp.ndarray) -> jnp.ndarray:
        """The actual schedule function passed to Optax."""
        step = jnp.asarray(count, dtype=jnp.float32)

        # Find the index of the active rule. The active rule is the one with the
        # largest starting_step that is less than or equal to the current step.
        eligible_mask = step >= start_steps
        # Replace non-eligible start_steps with a large negative number so they
        # are ignored by argmax.
        effective_starts = jnp.where(eligible_mask, start_steps, -1.0)
        active_rule_idx = jnp.argmax(effective_starts)

        # Evaluate the active rule using jax.lax.switch for efficient branching.
        lr = jax.lax.switch(active_rule_idx, rule_fns, step)

        # If the current step is before any rule starts, use the pre-start LR.
        return jnp.where(step < min_start_step, pre_start_lr, lr)

    return schedule
