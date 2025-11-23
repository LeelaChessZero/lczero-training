from typing import Dict, Protocol, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import xlogy

from lczero_training.training.state import TrainingSample
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
        self.policy_losses: Dict[str, PolicyLoss] = {
            loss_config.name: PolicyLoss(loss_config)
            for loss_config in config.policy
        }
        self.policy_weights: Dict[str, float] = {
            loss_config.name: loss_config.weight
            for loss_config in config.policy
        }
        self.value_losses: Dict[str, ValueLoss] = {
            loss_config.name: ValueLoss() for loss_config in config.value
        }
        self.value_weights: Dict[str, float] = {
            loss_config.name: loss_config.weight for loss_config in config.value
        }
        self.movesleft_losses: Dict[str, MovesLeftLoss] = {
            loss_config.name: MovesLeftLoss()
            for loss_config in config.movesleft
        }
        self.movesleft_weights: Dict[str, float] = {
            loss_config.name: loss_config.weight
            for loss_config in config.movesleft
        }

    def __call__(
        self,
        model: LczeroModel,
        sample: TrainingSample,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Run model forward pass.
        value_preds, policy_preds, movesleft_preds = model(sample.inputs)

        unweighted_losses = {}
        weighted_losses = []

        # Policy losses - pass entire sample.
        for name, policy_loss_fn in self.policy_losses.items():
            loss = policy_loss_fn(policy_preds[name], sample)
            unweighted_losses[f"policy/{name}"] = loss
            weighted_losses.append(loss * self.policy_weights[name])

        # Value losses - pass entire sample (WDL conversion happens in ValueLoss).
        for name, value_loss_fn in self.value_losses.items():
            loss = value_loss_fn(value_preds[name], sample)
            unweighted_losses[f"value/{name}"] = loss
            weighted_losses.append(loss * self.value_weights[name])

        # Movesleft losses - pass entire sample.
        for name, movesleft_loss_fn in self.movesleft_losses.items():
            loss = movesleft_loss_fn(movesleft_preds[name], sample)
            unweighted_losses[f"movesleft/{name}"] = loss
            weighted_losses.append(loss * self.movesleft_weights[name])

        data_loss = jnp.sum(jnp.array(weighted_losses))

        return data_loss, unweighted_losses


class ValueLoss:
    def __call__(
        self,
        value_pred: jax.Array,
        sample: TrainingSample,
    ) -> jax.Array:
        # Extract raw q/d from sample and compute WDL.
        # sample.values shape: [6, 3] where index 0 is result.
        result_q = sample.values[0, 0]
        result_d = sample.values[0, 1]
        # Compute WDL: w = (1 + q - d) / 2, l = (1 - q - d) / 2
        result_w = (1.0 + result_q - result_d) / 2.0
        result_l = (1.0 - result_q - result_d) / 2.0
        result_wdl = jnp.stack([result_w, result_d, result_l], axis=-1)

        # The cross-entropy between the predicted value and the target value.
        value_cross_entropy = optax.softmax_cross_entropy(
            logits=value_pred, labels=jax.lax.stop_gradient(result_wdl)
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
        sample: TrainingSample,
    ) -> jax.Array:
        # Extract probabilities from sample.
        policy_targets = jnp.asarray(
            sample.probabilities, dtype=policy_pred.dtype
        )
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
        sample: TrainingSample,
    ) -> jax.Array:
        # Extract movesleft from sample.
        # sample.values shape: [6, 3] where index 0 is result, component 2 is movesleft.
        movesleft_targets = sample.values[0, 2]

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
