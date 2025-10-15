import math
from functools import partial

import jax.numpy as jnp
import optax
from flax import nnx

from proto.training_config_pb2 import (
    LrSchedule,
    NadamwOptimizerConfig,
    OptimizerConfig,
)


def _interp(start_lr: float, end_lr: float, t: float, kind: int) -> float:
    if kind == LrSchedule.Transition.LINEAR:
        return start_lr + (end_lr - start_lr) * t
    if kind == LrSchedule.Transition.COSINE:
        return start_lr + 0.5 * (1.0 - math.cos(math.pi * t)) * (
            end_lr - start_lr
        )
    return start_lr


def _pick_rule(rules: list[LrSchedule], step: float) -> LrSchedule:
    eligible = (r for r in rules if r.starting_step <= step)
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


def _make_nadamw_weight_decay_mask(
    config: NadamwOptimizerConfig, params: nnx.State
) -> nnx.State:
    """Creates a mask that excludes bias and LayerNorm parameters from decay."""

    def is_layer_norm(path: tuple[object, ...]) -> bool:
        return any(str(s).startswith("ln") for s in path)

    def is_embedding(path: tuple[object, ...]) -> bool:
        return ("embedding", "embedding") in zip(path, path[1:])

    def is_bias(path: tuple[object, ...]) -> bool:
        return str(path[-1]).lower() == "bias"

    def mask_fn(path: tuple[object, ...], variable: nnx.Variable) -> bool:
        if is_bias(path) and not config.decay_biases:
            return False
        if is_layer_norm(path) and not config.decay_layer_norms:
            return False
        if is_embedding(path) and not config.decay_embedding:
            return False
        return True

    return nnx.map_state(mask_fn, params)


def make_lr_schedule(config: OptimizerConfig) -> optax.Schedule:
    rules = list(config.lr_schedule)

    def schedule(count: jnp.ndarray) -> jnp.ndarray:
        step = float(jnp.asarray(count, dtype=jnp.float32))
        rule = _pick_rule(rules, step)
        return _eval_rule(rule, step)

    return schedule


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: float | None = None,
) -> optax.GradientTransformation:
    lr_schedule = make_lr_schedule(config)
    if config.HasField("nadamw"):
        conf = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=conf.beta_1,
            b2=conf.beta_2,
            eps=conf.epsilon,
            weight_decay=conf.weight_decay,
            mask=partial(_make_nadamw_weight_decay_mask, conf),
        )
        if max_grad_norm is not None and max_grad_norm > 0:
            tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
        return tx
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
