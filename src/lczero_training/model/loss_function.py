from typing import Dict, Protocol, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from jax.scipy.special import xlogy

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

        unweighted_losses = {
            "value": self.value_loss(value_pred, value_targets),
            "policy": self.policy_loss(policy_pred, policy_targets),
            "movesleft": self.movesleft_loss(movesleft_pred, movesleft_targets),
        }

        data_loss = tree_util.tree_reduce(
            jnp.add,
            tree_util.tree_map(jnp.multiply, self.weights, unweighted_losses),
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
        if config.type == PolicyLossWeightsConfig.LOSS_TYPE_UNSPECIFIED:
            raise ValueError(
                f"Policy loss type must be specified for head '{config.name}'."
            )
        self._loss_type = config.type
        temperature = config.temperature
        if temperature <= 0:
            temperature = 1.0
        self._temperature = temperature

    def __call__(
        self,
        policy_pred: jax.Array,
        policy_targets: jax.Array,
    ) -> jax.Array:
        policy_targets = jnp.asarray(policy_targets, dtype=policy_pred.dtype)
        if self.config.illegal_moves == PolicyLossWeightsConfig.MASK:
            policy_pred = jnp.where(policy_targets >= 0, policy_pred, -jnp.inf)

        # Zero out negative targets for illegal moves.
        policy_targets = jax.nn.relu(policy_targets)

        if self._loss_type == PolicyLossWeightsConfig.KL:
            loss = self._kl_loss(policy_pred, policy_targets)
        elif self._loss_type == PolicyLossWeightsConfig.CROSS_ENTROPY:
            loss = self._cross_entropy_loss(policy_pred, policy_targets)
        else:
            raise AssertionError(
                f"Unsupported policy loss type: {self._loss_type}."
            )
        assert isinstance(loss, jax.Array)
        return loss

    def _cross_entropy_loss(
        self, policy_pred: jax.Array, policy_targets: jax.Array
    ) -> jax.Array:
        # Safe softmax cross-entropy to avoid NaNs due to -inf in logits.
        return optax.safe_softmax_cross_entropy(
            logits=policy_pred, labels=jax.lax.stop_gradient(policy_targets)
        )

    def _kl_loss(
        self, policy_pred: jax.Array, policy_targets: jax.Array
    ) -> jax.Array:
        if self._temperature != 1.0:
            policy_targets = jnp.power(policy_targets, 1.0 / self._temperature)

        target_sum = jnp.sum(policy_targets, axis=-1, keepdims=True)
        safe_sum = jnp.where(
            target_sum > 0, target_sum, jnp.ones_like(target_sum)
        )
        normalized_targets = policy_targets / safe_sum
        safe_targets = jax.lax.stop_gradient(normalized_targets)

        cross_entropy = optax.safe_softmax_cross_entropy(
            logits=policy_pred, labels=safe_targets
        )
        target_entropy = -jnp.sum(
            xlogy(normalized_targets, normalized_targets), axis=-1
        )
        return cross_entropy - target_entropy


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
