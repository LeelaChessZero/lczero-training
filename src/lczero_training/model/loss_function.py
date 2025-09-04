from typing import Dict, Protocol, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from proto.training_config_pb2 import LossWeightsConfig

from .model import LczeroModel


class Named(Protocol):
    name: str


def _find_head(field: Sequence[Named], name: str) -> Named:
    return next(head for head in field if head.name == name)


class LczeroLoss(nnx.Module):
    def __init__(self, model: LczeroModel, config: LossWeightsConfig):
        self.model = model
        self.config = config

        self.weights = {
            "policy": _find_head(config.policy, "main"),
            "value": _find_head(config.value, "winner"),
            "movesleft": _find_head(config.movesleft, "main"),
            "l2": config.l2_weight,
        }

        self.policy_loss = PolicyLoss()
        self.value_loss = ValueLoss()
        self.movesleft_loss = MovesLeftLoss()

    def __call__(
        self,
        inputs: jax.Array,
        value_targets: jax.Array,
        policy_targets: jax.Array,
        movesleft_targets: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        value_pred, policy_pred, movesleft_pred = self.model(inputs)

        l2_loss = optax.l2_loss(nnx.state(self.model, nnx.Param))

        unweighted_losses = {
            "value": self.value_loss(value_pred, value_targets),
            "policy": self.policy_loss(policy_pred, policy_targets),
            "movesleft": self.movesleft_loss(movesleft_pred, movesleft_targets),
            "l2": tree_util.tree_reduce(jnp.add, l2_loss),
        }

        total_loss = tree_util.tree_reduce(
            jnp.add,
            tree_util.tree_map(jnp.multiply, self.weights, unweighted_losses),
        )

        return total_loss, unweighted_losses


class ValueLoss(nnx.Module):
    def __init__(self) -> None:
        pass

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


class PolicyLoss(nnx.Module):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        policy_pred: jax.Array,
        policy_targets: jax.Array,
    ) -> jax.Array:
        # The cross-entropy between the predicted policy and the target policy.
        # This is equivalent to the KL divergence between the target and
        # predicted policies.
        policy_cross_entropy = optax.softmax_cross_entropy(
            logits=policy_pred, labels=jax.lax.stop_gradient(policy_targets)
        )

        # The entropy of the target policy.
        target_entropy = -jnp.sum(
            policy_targets * jnp.log(policy_targets + 1e-8), axis=-1
        )

        # The KL divergence is the cross-entropy minus the entropy.
        policy_kld = policy_cross_entropy - target_entropy
        assert isinstance(policy_kld, jax.Array)
        return policy_kld


class MovesLeftLoss(nnx.Module):
    def __init__(self) -> None:
        pass

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
            predictions=predictions, labels=targets, delta=10.0 / scale
        )
        assert isinstance(huber_loss, jax.Array)
        return huber_loss
