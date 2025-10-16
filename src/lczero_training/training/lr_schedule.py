import math
from typing import List, Sequence

import jax.numpy as jnp
import optax

from proto.training_config_pb2 import LrSchedule


def _interp(start_lr: float, end_lr: float, t: float, kind: int) -> float:
    if kind == LrSchedule.Transition.LINEAR:
        return start_lr + (end_lr - start_lr) * t
    if kind == LrSchedule.Transition.COSINE:
        return start_lr + 0.5 * (1.0 - math.cos(math.pi * t)) * (
            end_lr - start_lr
        )
    return start_lr


def _pick_rule(rules: List[LrSchedule], step: float) -> LrSchedule:
    eligible = [r for r in rules if r.starting_step <= step]
    if not eligible:
        raise ValueError(
            "No learning rate schedule is eligible for the given step."
        )
    return max(eligible, key=lambda r: r.starting_step)


def _eval_rule(rule: LrSchedule, step: float) -> jnp.ndarray:
    rel = step - rule.starting_step
    lrs = rule.lr

    period = sum(rule.duration_steps)
    if not rule.loop and rel >= period:
        return jnp.asarray(lrs[-1], dtype=jnp.float32)
    rel = rel % period

    acc = 0.0
    for i, d in enumerate(rule.duration_steps):
        if d == 0:
            continue
        if rel < acc + d:
            a = lrs[i]
            k = (
                rule.transition[i]
                if i < len(rule.transition)
                else LrSchedule.Transition.CONSTANT
            )
            t = max(0.0, min(1.0, (rel - acc) / d))
            b = lrs[i + 1] if i + 1 < len(lrs) else a
            return jnp.asarray(_interp(a, b, t, k), dtype=jnp.float32)
        acc += d

    assert False, "Unreachable: rel did not fall into any duration interval."


def make_lr_schedule(schedules: Sequence[LrSchedule]) -> optax.Schedule:
    rules = list(schedules)

    def schedule(count: jnp.ndarray) -> jnp.ndarray:
        step = float(jnp.asarray(count, dtype=jnp.float32))
        rule = _pick_rule(rules, step)
        return _eval_rule(rule, step)

    return schedule
