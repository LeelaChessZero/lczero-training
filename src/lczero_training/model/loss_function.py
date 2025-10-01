from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import xlogy

from proto.training_config_pb2 import (
    LossWeightsConfig,
    PolicyKLLossWeightsConfig,
    PolicyLossWeightsConfig,
)

from .model import LczeroModel


@dataclass
class _WeightedLoss:
    """Callable loss along with the metric key and scalar weight.

    The callable must accept the model predictions and targets for a given head
    and return a per-example loss array whose shape matches the batch
    dimensions of the inputs.
    """

    key: str
    weight: float
    fn: Callable[[jax.Array, jax.Array], jax.Array]


class LczeroLoss:
    def __init__(self, config: LossWeightsConfig):
        self.config = config
        self._policy_losses: List[_WeightedLoss] = [
            _WeightedLoss(
                key=f"policy_crossentropy_{cfg.name}",
                weight=cfg.weight,
                fn=PolicyCrossEntropyLoss(cfg),
            )
            for cfg in config.policy_crossentropy
        ]
        self._policy_losses.extend(
            _WeightedLoss(
                key=f"policy_kl_{cfg.name}",
                weight=cfg.weight,
                fn=PolicyKLLoss(cfg),
            )
            for cfg in config.policy_kl
        )

        self._value_losses: List[_WeightedLoss] = [
            _WeightedLoss(
                key=f"value_{cfg.name}",
                weight=cfg.weight,
                fn=ValueLoss(),
            )
            for cfg in config.value
        ]

        self._movesleft_losses: List[_WeightedLoss] = [
            _WeightedLoss(
                key=f"movesleft_{cfg.name}",
                weight=cfg.weight,
                fn=MovesLeftLoss(),
            )
            for cfg in config.movesleft
        ]

    def __call__(
        self,
        model: LczeroModel,
        inputs: jax.Array,
        value_targets: jax.Array,
        policy_targets: jax.Array,
        movesleft_targets: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        value_pred, policy_pred, movesleft_pred = model(inputs)

        weighted_losses: List[jax.Array] = []
        unweighted_losses: Dict[str, jax.Array] = {}

        def accumulate(
            specs: Sequence[_WeightedLoss],
            predictions: jax.Array,
            targets: jax.Array,
        ) -> None:
            for spec in specs:
                loss = spec.fn(predictions, targets)
                unweighted_losses[spec.key] = loss
                weighted_losses.append(spec.weight * loss)

        accumulate(self._policy_losses, policy_pred, policy_targets)
        accumulate(self._value_losses, value_pred, value_targets)
        accumulate(self._movesleft_losses, movesleft_pred, movesleft_targets)

        data_loss = jnp.sum(jnp.stack(weighted_losses, axis=0), axis=0)

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


class PolicyCrossEntropyLoss:
    def __init__(self, config: PolicyLossWeightsConfig):
        self.config = config

    def __call__(
        self,
        policy_pred: jax.Array,
        policy_targets: jax.Array,
    ) -> jax.Array:
        if self.config.illegal_moves == PolicyLossWeightsConfig.MASK:
            policy_pred = jnp.where(policy_targets >= 0, policy_pred, -jnp.inf)

        # Zero out negative targets for illegal moves.
        policy_targets = jax.nn.relu(policy_targets)

        # Safe softmax cross-entropy to avoid NaNs due to -inf in logits.
        loss = optax.safe_softmax_cross_entropy(
            logits=policy_pred, labels=jax.lax.stop_gradient(policy_targets)
        )
        assert isinstance(loss, jax.Array)
        return loss


class PolicyKLLoss:
    def __init__(self, config: PolicyKLLossWeightsConfig):
        self.config = config
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

        policy_targets = jax.nn.relu(policy_targets)

        if self._temperature != 1.0:
            policy_targets = jnp.power(policy_targets, 1.0 / self._temperature)

        target_sum = jnp.sum(policy_targets, axis=-1, keepdims=True)
        safe_sum = jnp.where(
            target_sum > 0, target_sum, jnp.ones_like(target_sum)
        )
        policy_targets = policy_targets / safe_sum

        safe_targets = jax.lax.stop_gradient(policy_targets)

        cross_entropy = optax.safe_softmax_cross_entropy(
            logits=policy_pred, labels=safe_targets
        )
        target_entropy = -jnp.sum(
            xlogy(policy_targets, policy_targets), axis=-1
        )
        loss = cross_entropy - target_entropy
        assert isinstance(loss, jax.Array)
        return loss


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
