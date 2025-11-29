from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import xlogy

from lczero_training.training.state import TrainingSample
from proto.training_config_pb2 import (
    LossWeightsConfig,
    MovesLeftLossWeightsConfig,
    PolicyLossWeightsConfig,
    ValueCategoricalLossWeightsConfig,
    ValueErrorLossWeightsConfig,
    ValueLossWeightsConfig,
)

from .model import LczeroModel, ModelPrediction


def _compute_q_from_wdl(wdl_logits: jax.Array) -> jax.Array:
    """Compute Q value from WDL logits."""
    wdl_probs = jax.nn.softmax(wdl_logits)
    q_weights = jnp.array([1.0, 0.0, -1.0])
    return jnp.dot(wdl_probs, q_weights)


class LossBase:
    def __init__(
        self,
        config: Union[
            PolicyLossWeightsConfig,
            ValueLossWeightsConfig,
            MovesLeftLossWeightsConfig,
            ValueErrorLossWeightsConfig,
            ValueCategoricalLossWeightsConfig,
        ],
    ) -> None:
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        raise NotImplementedError("Subclasses must implement __call__")


class LczeroLoss:
    policy_losses: List["PolicyLoss"]
    value_losses: List["ValueLoss"]
    movesleft_losses: List["MovesLeftLoss"]
    value_error_losses: List["ValueErrorLoss"]
    value_categorical_losses: List["ValueCategoricalLoss"]

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
        self.value_error_losses = [
            ValueErrorLoss(loss_config) for loss_config in config.value_error
        ]
        self.value_categorical_losses = [
            ValueCategoricalLoss(loss_config)
            for loss_config in config.value_categorical
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
        _validate_no_duplicate_metrics("value_error", self.value_error_losses)
        _validate_no_duplicate_metrics(
            "value_categorical", self.value_categorical_losses
        )

    def __call__(
        self,
        model: LczeroModel,
        sample: TrainingSample,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Run model forward pass.
        predictions = model(sample.inputs)

        unweighted_losses: Dict[str, jax.Array] = {}
        weighted_losses: List[jax.Array] = []

        for policy_loss in self.policy_losses:
            loss = policy_loss(predictions, sample)
            unweighted_losses[f"policy/{policy_loss.metric_name}"] = loss
            weighted_losses.append(loss * policy_loss.weight)

        for value_loss in self.value_losses:
            loss = value_loss(predictions, sample)
            unweighted_losses[f"value/{value_loss.metric_name}"] = loss
            weighted_losses.append(loss * value_loss.weight)

        for movesleft_loss in self.movesleft_losses:
            loss = movesleft_loss(predictions, sample)
            unweighted_losses[f"movesleft/{movesleft_loss.metric_name}"] = loss
            weighted_losses.append(loss * movesleft_loss.weight)

        for value_error_loss in self.value_error_losses:
            loss = value_error_loss(predictions, sample)
            unweighted_losses[f"value_error/{value_error_loss.metric_name}"] = (
                loss
            )
            weighted_losses.append(loss * value_error_loss.weight)

        for value_categorical_loss in self.value_categorical_losses:
            loss = value_categorical_loss(predictions, sample)
            unweighted_losses[
                f"value_categorical/{value_categorical_loss.metric_name}"
            ] = loss
            weighted_losses.append(loss * value_categorical_loss.weight)

        data_loss = jnp.sum(jnp.array(weighted_losses))

        return data_loss, unweighted_losses


class ValueLoss(LossBase):
    def __init__(self, config: ValueLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        value_logits = value_pred[0]
        # Extract raw q/d from sample and compute WDL.
        value_q = sample.values[self.value_type, 0]
        value_d = sample.values[self.value_type, 1]
        # Compute WDL: w = (1 + q - d) / 2, l = (1 - q - d) / 2
        value_w = (1.0 + value_q - value_d) / 2.0
        value_l = (1.0 - value_q - value_d) / 2.0
        value_wdl = jnp.stack([value_w, value_d, value_l], axis=-1)

        # The cross-entropy between the predicted value and the target value.
        value_cross_entropy = optax.softmax_cross_entropy(
            logits=value_logits, labels=jax.lax.stop_gradient(value_wdl)
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

        # Store optimistic config if present.
        if config.HasField("optimistic"):
            opt = config.optimistic
            self.opt_value_head: Optional[str] = opt.value_head_name
            self.opt_value_type = opt.value_type
            self.opt_strength = opt.strength
            self.opt_eps = opt.eps
            self.opt_alpha = opt.alpha
            self.opt_propagate_gradients = opt.propagate_value_gradients
        else:
            self.opt_value_head = None

    def _apply_temperature_and_normalize(
        self, policy_targets: jax.Array
    ) -> jax.Array:
        if self._temperature == 1.0:
            return policy_targets

        # Apply temperature scaling.
        policy_targets = jnp.power(policy_targets, 1.0 / self._temperature)

        # Renormalize after temperature scaling.
        target_sum = jnp.sum(policy_targets, axis=-1, keepdims=True)
        safe_sum = jnp.where(
            target_sum > 0, target_sum, jnp.ones_like(target_sum)
        )
        return policy_targets / safe_sum

    def _compute_optimistic_weight(
        self,
        value_pred: Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]],
        target_q: jax.Array,
    ) -> jax.Array:
        """Compute optimistic policy weight from value head predictions."""
        wdl_logits = value_pred[0]
        error_pred = value_pred[1]
        assert error_pred is not None, (
            "Error prediction required for optimistic weighting"
        )

        # Optionally block gradients to value and error heads.
        if not self.opt_propagate_gradients:
            wdl_logits = jax.lax.stop_gradient(wdl_logits)
            error_pred = jax.lax.stop_gradient(error_pred)

        # Compute predicted Q from WDL.
        q_pred = _compute_q_from_wdl(wdl_logits)

        # Compute sigma and z-score.
        sigma = jnp.sqrt(error_pred.squeeze())
        z = (target_q - q_pred) / (sigma + self.opt_eps)

        # Compute weight.
        return jax.nn.sigmoid((z - self.opt_strength) * self.opt_alpha)

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        policy_pred = predictions.policy[self.head_name]
        # Extract probabilities from sample.
        policy_targets = jnp.asarray(
            sample.probabilities, dtype=policy_pred.dtype
        )
        if self.config.illegal_moves == PolicyLossWeightsConfig.MASK:
            policy_pred = jnp.where(policy_targets >= 0, policy_pred, -jnp.inf)

        # Zero out negative targets for illegal moves.
        policy_targets = jax.nn.relu(policy_targets)

        # Apply temperature scaling and renormalization if needed.
        policy_targets = self._apply_temperature_and_normalize(policy_targets)

        cross_entropy = cast(
            jax.Array,
            optax.safe_softmax_cross_entropy(
                logits=policy_pred, labels=policy_targets
            ),
        )
        if self._loss_type == PolicyLossWeightsConfig.CROSS_ENTROPY:
            loss = cross_entropy
        elif self._loss_type == PolicyLossWeightsConfig.KL:
            loss = cross_entropy + jnp.sum(
                xlogy(policy_targets, policy_targets), axis=-1
            )
        else:
            raise AssertionError(
                f"Unknown policy loss type: {self._loss_type}."
            )

        # Apply optimistic weighting if configured.
        if self.opt_value_head is not None:
            value_pred = predictions.value[self.opt_value_head]
            target_q = sample.values[self.opt_value_type, 0]
            loss = loss * self._compute_optimistic_weight(value_pred, target_q)

        return loss


class MovesLeftLoss(LossBase):
    def __init__(self, config: MovesLeftLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        movesleft_pred = predictions.movesleft[self.head_name]
        # Extract movesleft from sample.
        # sample.values shape: [6, 3], component 2 is movesleft.
        movesleft_targets = sample.values[self.value_type, 2]

        # Scale the loss to similar range as other losses.
        scale = 20.0
        targets = movesleft_targets / scale
        scaled_predictions = movesleft_pred / scale

        # Huber loss
        huber_loss = optax.huber_loss(
            predictions=scaled_predictions, targets=targets, delta=10.0 / scale
        )
        assert isinstance(huber_loss, jax.Array)
        return huber_loss.squeeze()


class ValueErrorLoss(LossBase):
    def __init__(self, config: ValueErrorLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type
        self.propagate_value_gradients = config.propagate_value_gradients

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        wdl_logits = value_pred[0]
        error_pred = value_pred[1]

        # Convert WDL to Q value.
        predicted_q = _compute_q_from_wdl(wdl_logits)

        # Get target Q value.
        target_q = sample.values[self.value_type, 0]

        # Compute actual squared error.
        actual_squared_error = jnp.square(predicted_q - target_q)

        # Optionally block gradients to WDL head.
        if not self.propagate_value_gradients:
            actual_squared_error = jax.lax.stop_gradient(actual_squared_error)

        # MSE between error prediction and actual error.
        loss = jnp.square(error_pred - actual_squared_error)

        return loss.squeeze()


class ValueCategoricalLoss(LossBase):
    def __init__(self, config: ValueCategoricalLossWeightsConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        categorical_logits = value_pred[2]
        assert categorical_logits is not None

        # Get target Q value from sample.
        target_q = sample.values[self.value_type, 0]

        # Convert Q to bucket index: map [-1, 1) to [0, num_buckets).
        num_buckets = categorical_logits.shape[-1]
        bucket_index = jnp.floor((target_q + 1.0) / 2.0 * num_buckets).astype(
            jnp.int32
        )
        bucket_index = jnp.clip(bucket_index, 0, num_buckets - 1)

        # Create one-hot target.
        target_one_hot = jax.nn.one_hot(bucket_index, num_buckets)

        # Compute softmax cross-entropy.
        loss = optax.softmax_cross_entropy(
            logits=categorical_logits,
            labels=jax.lax.stop_gradient(target_one_hot),
        )
        assert isinstance(loss, jax.Array)
        return loss
