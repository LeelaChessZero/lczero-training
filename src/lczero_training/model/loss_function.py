from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import xlogy

from lczero_training.training.state import TrainingSample
from proto.training_config_pb2 import (
    LossWeightsConfig,
    MovesLeftLossWeightsConfig,
    PolicyLossWeightsConfig,
    ValueLossWeightsConfig,
)

from .model import LczeroModel


class LossBase:
    def __init__(
        self,
        config: Union[
            PolicyLossWeightsConfig,
            ValueLossWeightsConfig,
            MovesLeftLossWeightsConfig,
        ],
    ) -> None:
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight

    def __call__(self, pred: jax.Array, sample: TrainingSample) -> jax.Array:
        raise NotImplementedError("Subclasses must implement __call__")


class LczeroLoss:
    policy_losses: List["PolicyLoss"]
    value_losses: List["ValueLoss"]
    movesleft_losses: List["MovesLeftLoss"]

    def __init__(self, config: LossWeightsConfig) -> None:
        self.config = config
        self.policy_losses = [
            PolicyLoss(loss_config) for loss_config in config.policy
        ]
        self.value_losses = [
            ValueLoss(loss_config) for loss_config in config.value
        ]
        self.movesleft_losses = [
            MovesLeftLoss(loss_config) for loss_config in config.movesleft
        ]

        def _validate_no_duplicate_metrics(
            loss_type_name: str, losses: Sequence[LossBase]
        ) -> None:
            seen = set()
            for name in (loss.metric_name for loss in losses):
                if name in seen:
                    raise ValueError(
                        f"Duplicate metric name: {loss_type_name}/{name}"
                    )
                seen.add(name)

        _validate_no_duplicate_metrics("policy", self.policy_losses)
        _validate_no_duplicate_metrics("value", self.value_losses)
        _validate_no_duplicate_metrics("movesleft", self.movesleft_losses)

    def __call__(
        self,
        model: LczeroModel,
        sample: TrainingSample,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Run model forward pass.
        value_preds, policy_preds, movesleft_preds = model(sample.inputs)

        unweighted_losses: Dict[str, jax.Array] = {}
        weighted_losses: List[jax.Array] = []

        loss_configs: List[
            Tuple[str, Dict[str, jax.Array], Sequence[LossBase]]
        ] = [
            ("policy", policy_preds, self.policy_losses),
            ("value", value_preds, self.value_losses),
            ("movesleft", movesleft_preds, self.movesleft_losses),
        ]

        for category_name, preds, losses in loss_configs:
            for loss_fn in losses:
                loss = loss_fn(preds[loss_fn.head_name], sample)
                unweighted_losses[f"{category_name}/{loss_fn.metric_name}"] = (
                    loss
                )
                weighted_losses.append(loss * loss_fn.weight)

        data_loss = jnp.sum(jnp.array(weighted_losses))

        return data_loss, unweighted_losses


class ValueLoss(LossBase):
    def __init__(self, config: ValueLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        value_pred: jax.Array,
        sample: TrainingSample,
    ) -> jax.Array:
        # Extract raw q/d from sample and compute WDL.
        value_q = sample.values[self.value_type, 0]
        value_d = sample.values[self.value_type, 1]
        # Compute WDL: w = (1 + q - d) / 2, l = (1 - q - d) / 2
        value_w = (1.0 + value_q - value_d) / 2.0
        value_l = (1.0 - value_q - value_d) / 2.0
        value_wdl = jnp.stack([value_w, value_d, value_l], axis=-1)

        # The cross-entropy between the predicted value and the target value.
        value_cross_entropy = optax.softmax_cross_entropy(
            logits=value_pred, labels=jax.lax.stop_gradient(value_wdl)
        )
        assert isinstance(value_cross_entropy, jax.Array)
        return value_cross_entropy


class PolicyLoss(LossBase):
    def __init__(self, config: PolicyLossWeightsConfig):
        super().__init__(config)
        self.config = config
        if config.type == PolicyLossWeightsConfig.LOSS_TYPE_UNSPECIFIED:
            raise ValueError(
                f"Policy loss type must be specified for head '{config.head_name}'."
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


class MovesLeftLoss(LossBase):
    def __init__(self, config: MovesLeftLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        movesleft_pred: jax.Array,
        sample: TrainingSample,
    ) -> jax.Array:
        # Extract movesleft from sample.
        # sample.values shape: [6, 3], component 2 is movesleft.
        movesleft_targets = sample.values[self.value_type, 2]

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
