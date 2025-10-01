import jax.numpy as jnp
import optax

from proto.training_config_pb2 import LinearWarmupLRSchedule, OptimizerConfig


def _make_linear_warmup_schedule(
    config: LinearWarmupLRSchedule,
) -> optax.Schedule:
    steps = jnp.asarray(config.step, dtype=jnp.float32)
    lrs = jnp.asarray(config.lr, dtype=jnp.float32)

    if steps.size != lrs.size:
        raise ValueError(
            "linear_warmup_lr.step and lr must have the same length"
        )
    if steps.size == 0:
        raise ValueError("linear_warmup_lr requires at least one step")
    if jnp.any(steps[1:] < steps[:-1]):
        raise ValueError("linear_warmup_lr.step must be sorted ascending")

    def schedule(count: jnp.ndarray) -> jnp.ndarray:
        step = jnp.asarray(count, dtype=jnp.float32)

        lower = jnp.searchsorted(steps, step, side="right") - 1
        lower = jnp.clip(lower, 0, steps.size - 1)
        upper = jnp.clip(lower + 1, 0, steps.size - 1)

        left_step = steps[lower]
        right_step = steps[upper]
        left_lr = lrs[lower]
        right_lr = lrs[upper]

        progress = jnp.where(
            right_step == left_step,
            0.0,
            (step - left_step) / (right_step - left_step),
        )
        progress = jnp.clip(progress, 0.0, 1.0)
        return left_lr + progress * (right_lr - left_lr)

    return schedule


def make_lr_schedule(config: OptimizerConfig) -> optax.Schedule:
    if config.HasField("constant_lr"):
        return optax.constant_schedule(config.constant_lr.lr)
    elif config.HasField("linear_warmup_lr"):
        return _make_linear_warmup_schedule(config.linear_warmup_lr)
    else:
        raise ValueError(
            "Unsupported learning rate schedule: {}".format(
                config.WhichOneof("lr_schedule")
            )
        )


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
