from functools import partial

import jax
import jax.numpy as jnp
import optax

from lczero_training.training.utils import make_weights_mask
from proto.training_config_pb2 import OptimizerConfig


def update_optimizer_step(
    opt_state: optax.OptState, step: int
) -> optax.OptState:
    """Updates all step counters in the optimizer state tree."""
    step_array = jnp.array(step, dtype=jnp.int32)

    def update_count(x: optax.OptState) -> optax.OptState:
        if isinstance(
            x,
            (
                optax.ScaleByAdamState,
                optax.ScaleByScheduleState,
            ),
        ):
            return x._replace(count=step_array)
        return x

    return jax.tree_util.tree_map(
        update_count, opt_state, is_leaf=lambda x: hasattr(x, "_replace")
    )


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: float | None = None,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    if config.HasField("nadamw"):
        conf = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=conf.beta_1,
            b2=conf.beta_2,
            eps=conf.epsilon,
            weight_decay=conf.weight_decay,
            mask=partial(make_weights_mask, conf.decay_selector),
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
