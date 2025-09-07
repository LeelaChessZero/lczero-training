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
) -> optax.GradientTransformation:
    lr_schedule = make_lr_schedule(config)
    if config.HasField("nadam"):
        conf = config.nadam
        return optax.adamw(
            lr_schedule, b1=conf.beta_1, b2=conf.beta_2, eps=conf.epsilon
        )
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
