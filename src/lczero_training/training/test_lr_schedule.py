from typing import Callable, List

import jax.numpy as jnp
import pytest

from lczero_training.training.lr_schedule import make_lr_schedule
from proto import training_config_pb2 as pb


def _sched(
    schedules: List[pb.LrSchedule],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return make_lr_schedule(schedules)


def _val(s: Callable[[jnp.ndarray], jnp.ndarray], t: int | float) -> float:
    return float(s(jnp.asarray(t, dtype=jnp.float32)))


def test_rule_selection_by_starting_step() -> None:
    r0 = pb.LrSchedule(
        starting_step=0,
        duration_steps=[5],
        lr=[0.1, 0.2],
    )
    r1 = pb.LrSchedule(
        starting_step=10,
        duration_steps=[5],
        lr=[0.5, 0.5],
    )
    sched = _sched([r0, r1])
    assert _val(sched, 0) == pytest.approx(0.1)
    assert _val(sched, 9) == pytest.approx(0.2)
    assert _val(sched, 10) == pytest.approx(0.5)
    assert _val(sched, 100) == pytest.approx(0.5)


def test_default_constant_transition_and_tail() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[5],
        lr=[0.3, 0.8],  # no transition specified -> CONSTANT
    )
    sched = _sched([r])
    for t in range(5):
        assert _val(sched, t) == pytest.approx(0.3)
    # Beyond period (no loop) yields last lr
    assert _val(sched, 6) == pytest.approx(0.8)


def test_linear_then_hold() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[3, 7],
        lr=[0.0, 0.9, 0.9],
        transition=[pb.LrSchedule.Transition.LINEAR],
    )
    sched = _sched([r])
    assert _val(sched, 0) == pytest.approx(0.0)
    assert _val(sched, 1) == pytest.approx(0.3)
    assert _val(sched, 2) == pytest.approx(0.6)
    assert _val(sched, 3) == pytest.approx(0.9)
    assert _val(sched, 8) == pytest.approx(0.9)


def test_looping_constant_segments() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[3, 2],
        lr=[1.0, 2.0, 3.0],
        loop=True,
    )
    sched = _sched([r])
    assert _val(sched, 0) == pytest.approx(1.0)
    assert _val(sched, 2) == pytest.approx(1.0)
    assert _val(sched, 3) == pytest.approx(2.0)
    assert _val(sched, 5) == pytest.approx(1.0)  # 5 % (3+2) == 0


def test_zero_duration_is_skipped() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[0, 5],
        lr=[1.0, 2.0, 3.0],
        transition=[
            pb.LrSchedule.Transition.LINEAR,
            pb.LrSchedule.Transition.LINEAR,
        ],
    )
    sched = _sched([r])
    # Interpolates over the second interval [2.0 -> 3.0]
    assert _val(sched, 0) == pytest.approx(2.0)
    assert _val(sched, 2) == pytest.approx(2.4)
    assert _val(sched, 5) == pytest.approx(3.0)


def test_chain_zero_durations_then_linear() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[0, 0, 0, 5],
        lr=[1.0, 2.0, 3.0, 4.0, 5.0],
        transition=[
            pb.LrSchedule.Transition.LINEAR,
            pb.LrSchedule.Transition.LINEAR,
            pb.LrSchedule.Transition.LINEAR,
            pb.LrSchedule.Transition.LINEAR,
        ],
    )
    sched = _sched([r])
    # Should use last interval [4.0 -> 5.0] linearly across 5 steps
    assert _val(sched, 0) == pytest.approx(4.0)
    assert _val(sched, 2) == pytest.approx(4.4)
    assert _val(sched, 4) == pytest.approx(4.8)


def test_cosine() -> None:
    r = pb.LrSchedule(
        starting_step=0,
        duration_steps=[4],
        lr=[0.0, 1.0],
        transition=[pb.LrSchedule.Transition.COSINE],
    )
    sched = _sched([r])
    # t=0 -> 0.0, t=0.5 -> 0.5, t=1.0 -> 1.0 approximately
    assert _val(sched, 0) == pytest.approx(0.0)
    assert _val(sched, 2) == pytest.approx(0.5, abs=1e-6)
    assert _val(sched, 4) == pytest.approx(1.0)


def test_before_first_rule_uses_earliest_first_lr() -> None:
    r = pb.LrSchedule(
        starting_step=5,
        duration_steps=[3],
        lr=[0.1, 0.2],
    )
    sched = _sched([r])
    # Before the first rule starts, schedule returns the first lr of earliest rule.
    assert _val(sched, 0) == pytest.approx(0.1)
