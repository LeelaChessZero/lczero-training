from typing import Optional

import optax

from proto.training_config_pb2 import OptimizerConfig


def make_lr_schedule(config: OptimizerConfig) -> optax.Schedule:
    if config.HasField("constant_lr"):
        return optax.constant_schedule(config.constant_lr.lr)
    else:
        raise ValueError(
            "Unsupported learning rate schedule: {}".format(
                config.WhichOneof("lr_schedule")
            )
        )


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: Optional[float] = None,
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
