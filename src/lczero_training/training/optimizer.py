from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from lczero_training.training.utils import make_weights_mask
from proto.training_config_pb2 import OptimizerConfig

_STATES_WITH_COUNT = (
    optax.ScaleByAdamState,
    optax.ScaleByScheduleState,
)


def update_optimizer_step(
    opt_state: optax.OptState, step: int
) -> optax.OptState:
    """Updates all step counters in the optimizer state tree."""
    step_array = jnp.array(step, dtype=jnp.int32)

    def is_known_state(x: object) -> bool:
        return isinstance(x, _STATES_WITH_COUNT)

    def update_count(x: optax.OptState) -> optax.OptState:
        return x._replace(count=step_array)

    result = jax.tree_util.tree_map(
        update_count, opt_state, is_leaf=is_known_state
    )

    # Verify no count fields were missed due to unknown wrapper types.
    def has_unexpected_count(x: object) -> bool:
        return (
            hasattr(x, "_fields")
            and "count" in x._fields
            and not isinstance(x, _STATES_WITH_COUNT)
        )

    unexpected = jax.tree.leaves(result, is_leaf=has_unexpected_count)
    assert not unexpected, (
        f"Unexpected state type(s) with 'count' field: "
        f"{[type(x).__name__ for x in unexpected]}"
    )

    return result


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: float | None = None,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    if config.HasField("nadamw"):
        nadamw = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=nadamw.beta_1,
            b2=nadamw.beta_2,
            eps=nadamw.epsilon,
            weight_decay=nadamw.weight_decay,
            mask=partial(make_weights_mask, nadamw.decay_selector),
        )
    elif config.HasField("nadam"):
        nadam = config.nadam
        tx = optax.nadam(
            lr_schedule,
            b1=nadam.beta_1,
            b2=nadam.beta_2,
            eps=nadam.epsilon,
        )
    elif config.HasField("sgd"):
        sgd = config.sgd
        tx = optax.sgd(
            lr_schedule,
            momentum=sgd.momentum if sgd.momentum else None,
            nesterov=sgd.nesterov,
        )
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
    if max_grad_norm is not None and max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    if config.HasField("freeze_selector"):
        freeze_mask = partial(make_weights_mask, config.freeze_selector)

        def trainable_mask(p: nnx.State) -> nnx.State:
            return jax.tree.map(lambda x: not x, freeze_mask(p))

        tx = optax.chain(
            optax.masked(tx, trainable_mask),
            optax.masked(optax.set_to_zero(), freeze_mask),
        )
    return tx
