from typing import Dict, Protocol, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from proto.training_config_pb2 import (
    LossWeightsConfig,
    PolicyLossWeightsConfig,
)

from .model import LczeroModel


class Named(Protocol):
    name: str


T = TypeVar("T", bound=Named)


def _find_head(field: Sequence[T], name: str) -> T:
    return next(head for head in field if head.name == name)


class LczeroLoss:
    def __init__(self, config: LossWeightsConfig):
        self.config = config
        main_policy_config = _find_head(config.policy, "main")
        winner_value_config = _find_head(config.value, "winner")
        main_movesleft_config = _find_head(config.movesleft, "main")

        self.weights = {
            "policy": main_policy_config.weight,
            "value": winner_value_config.weight,
            "movesleft": main_movesleft_config.weight,
        }
        self.l2_weight = config.l2_weight

        self.policy_loss = PolicyLoss(main_policy_config)
        self.value_loss = ValueLoss()
        self.movesleft_loss = MovesLeftLoss()

    def __call__(
        self,
        model: LczeroModel,
        inputs: jax.Array,
        value_targets: jax.Array,
        policy_targets: jax.Array,
        movesleft_targets: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        value_pred, policy_pred, movesleft_pred = model(inputs)

        # L2 loss
        params = nnx.state(model, nnx.Param)
        # We only want to regularize kernels, not biases or batch norm parameters.
        kernels = [
            p
            for path, p in tree_util.tree_flatten_with_path(params)[0]
            if "kernel" in tree_util.keystr(path)
        ]
        l2_loss_val = 0.00005 * sum(jnp.sum(jnp.square(p)) for p in kernels)

        unweighted_losses = {
            "value": self.value_loss(value_pred, value_targets),
            "policy": self.policy_loss(policy_pred, policy_targets),
            "movesleft": self.movesleft_loss(movesleft_pred, movesleft_targets),
            "l2": jnp.asarray(l2_loss_val),
        }

        unweighted_data_losses = {
            k: v for k, v in unweighted_losses.items() if k != "l2"
        }

        data_loss = tree_util.tree_reduce(
            jnp.add,
            tree_util.tree_map(
                jnp.multiply, self.weights, unweighted_data_losses
            ),
        )

        return data_loss, unweighted_losses


class ValueLoss:
    def __call__(
        self,
        value_pred: jax.Array,
        value_targets: jax.Array,
    ) -> jax.Array:
        # The cross-entropy between the predicted value and the target value.
        value_cross_entropy = optax.softmax_cross_entropy(
            logits=value_pred, labels=jax.lax.stop_gradient(value_targets)
        )
        assert isinstance(value_cross_entropy, jax.Array)
        return value_cross_entropy


class PolicyLoss:
    def __init__(self, config: PolicyLossWeightsConfig):
        self.config = config

    def __call__(
        self,
        policy_pred: jax.Array,
        policy_targets: jax.Array,
    ) -> jax.Array:
        if self.config.illegal_moves == PolicyLossWeightsConfig.MASK:
            move_is_legal = policy_targets >= 0
            illegal_filler = jnp.full_like(policy_pred, -1e10)
            policy_pred = jnp.where(move_is_legal, policy_pred, illegal_filler)

        # The cross-entropy between the predicted policy and the target policy.
        policy_targets = jax.nn.relu(policy_targets)
        policy_cross_entropy = optax.softmax_cross_entropy(
            logits=policy_pred, labels=jax.lax.stop_gradient(policy_targets)
        )
        assert isinstance(policy_cross_entropy, jax.Array)
        return policy_cross_entropy


class MovesLeftLoss:
    def __call__(
        self,
        movesleft_pred: jax.Array,
        movesleft_targets: jax.Array,
    ) -> jax.Array:
        # Scale the loss to similar range as other losses.
        scale = 20.0
        targets = movesleft_targets / scale
        predictions = movesleft_pred / scale

        # Huber loss
        huber_loss = optax.huber_loss(
            predictions=predictions, targets=targets, delta=10.0 / scale
        )
        assert isinstance(huber_loss, jax.Array)
        return huber_loss.squeeze()
