import jax
import jax.numpy as jnp
import pytest

from lczero_training.model.loss_function import MovesLeftLoss, ValueLoss
from lczero_training.model.model import ModelPrediction
from lczero_training.training.state import TrainingSample
from proto import training_config_pb2 as tc

NAN = float("nan")


def _valid_movesleft_config() -> tc.MovesLeftLossConfig:
    return tc.MovesLeftLossConfig(
        head_name="progress",
        value_type=tc.RESULT,
        component=tc.MovesLeftLossConfig.PLIES_UNTIL_PROGRESS,
        weight=0.5,
        scale=0.1,
        huber_delta=5.0,
    )


def _sample(progress: float) -> TrainingSample:
    # values[6, 4]; RESULT row holds [q, d, m, plies_until_progress].
    values = jnp.zeros((6, 4))
    values = values.at[tc.RESULT].set(jnp.array([0.5, 0.2, 42.5, progress]))
    return TrainingSample(
        inputs=jnp.zeros((112, 8, 8)),
        probabilities=jnp.zeros((1858,)),
        values=values,
    )


def _movesleft_pred(value: float) -> ModelPrediction:
    return ModelPrediction(
        value={}, policy={}, movesleft={"progress": jnp.array([value])}
    )


@pytest.mark.parametrize("field", ["scale", "huber_delta"])
def test_movesleft_requires_positive_scaling(field: str) -> None:
    cfg = _valid_movesleft_config()
    setattr(cfg, field, 0.0)
    with pytest.raises(ValueError):
        MovesLeftLoss(cfg)


@pytest.mark.parametrize(
    "component",
    [
        tc.MovesLeftLossConfig.COMPONENT_Q,
        tc.MovesLeftLossConfig.COMPONENT_D,
    ],
)
def test_movesleft_rejects_q_d_component(component: int) -> None:
    cfg = _valid_movesleft_config()
    setattr(cfg, "component", component)
    with pytest.raises(ValueError):
        MovesLeftLoss(cfg)


def test_movesleft_component_selects_column() -> None:
    # RESULT row is [q=0.5, d=0.2, m=42.5, p=7]; a prediction matching the
    # selected column gives zero loss, proving component picks the column.
    pred = _movesleft_pred(7.0)
    sample = _sample(7.0)

    p_cfg = _valid_movesleft_config()  # PLIES_UNTIL_PROGRESS -> col 3 = 7
    assert float(MovesLeftLoss(p_cfg)(pred, sample)) == 0.0

    m_cfg = _valid_movesleft_config()
    m_cfg.component = tc.MovesLeftLossConfig.MOVES_LEFT  # col 2 = 42.5
    assert float(MovesLeftLoss(m_cfg)(pred, sample)) > 0.0


def test_movesleft_finite_target_has_loss() -> None:
    loss = MovesLeftLoss(_valid_movesleft_config())
    value = float(loss(_movesleft_pred(3.0), _sample(7.0)))
    assert value > 0.0


def test_movesleft_nan_target_is_masked_with_finite_grad() -> None:
    loss = MovesLeftLoss(_valid_movesleft_config())

    def loss_of_pred(pred: float) -> jax.Array:
        return loss(_movesleft_pred(pred), _sample(NAN))

    assert float(loss_of_pred(3.0)) == 0.0
    grad = jax.grad(loss_of_pred)(3.0)
    assert float(grad) == 0.0
    assert bool(jnp.isfinite(grad))


def test_value_loss_masks_nan_target() -> None:
    # ORIG q/d may be NaN; that sample must contribute zero with finite grad.
    loss = ValueLoss(
        tc.ValueLossConfig(head_name="winner", value_type=tc.RESULT)
    )

    def loss_of_logits(logits: jax.Array, progress: float) -> jax.Array:
        nan_q_d = (
            jnp.zeros((6, 4))
            .at[tc.RESULT]
            .set(jnp.array([NAN, NAN, 0.0, progress]))
        )
        sample = TrainingSample(
            inputs=jnp.zeros((112, 8, 8)),
            probabilities=jnp.zeros((1858,)),
            values=nan_q_d,
        )
        pred = ModelPrediction(
            value={"winner": (logits, None, None)}, policy={}, movesleft={}
        )
        return loss(pred, sample)

    logits = jnp.array([0.1, 0.2, 0.3])
    assert float(loss_of_logits(logits, 0.0)) == 0.0
    grad = jax.grad(loss_of_logits)(logits, 0.0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(jnp.abs(grad))) == 0.0
