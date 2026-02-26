from pathlib import PurePosixPath

from flax import nnx

from proto.training_config_pb2 import WeightsSelector


def make_weights_mask(
    selector: WeightsSelector, params: nnx.State
) -> nnx.State:
    """Creates a boolean mask based on WeightsSelector. True = include weight."""

    def mask_fn(path: tuple[object, ...], _variable: nnx.Variable) -> bool:
        p = PurePosixPath(*map(str, path))
        for rule in selector.rule:
            if p.full_match(rule.match):
                return rule.include
        return selector.otherwise_include

    return nnx.map_state(mask_fn, params)
