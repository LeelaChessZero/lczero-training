from flax import nnx

from proto.training_config_pb2 import WeightsSelector


def is_layer_norm(path: tuple[object, ...]) -> bool:
    return any(str(s).startswith("ln") for s in path)


def is_embedding(path: tuple[object, ...]) -> bool:
    return ("embedding", "embedding") in zip(path, path[1:])


def is_bias(path: tuple[object, ...]) -> bool:
    return str(path[-1]).lower() == "bias"


def is_policy_head(path: tuple[object, ...]) -> bool:
    return "policy_heads" in map(str, path)


def is_value_head(path: tuple[object, ...]) -> bool:
    return "value_heads" in map(str, path)


def is_movesleft_head(path: tuple[object, ...]) -> bool:
    return "movesleft_heads" in map(str, path)


def is_policy_embedding_shared(path: tuple[object, ...]) -> bool:
    return "policy_embedding_shared" in map(str, path)


def make_weights_mask(
    selector: WeightsSelector, params: nnx.State
) -> nnx.State:
    """Creates a boolean mask based on WeightsSelector. True = include weight.

    If a param matches multiple categories (e.g., a bias inside layer_norm),
    it's excluded if ANY matching category has selector=false.
    """

    def mask_fn(path: tuple[object, ...], _variable: nnx.Variable) -> bool:
        if is_bias(path) and not selector.biases:
            return False
        if is_layer_norm(path) and not selector.layer_norms:
            return False
        if is_embedding(path) and not selector.embedding:
            return False
        if (
            is_policy_head(path) or is_policy_embedding_shared(path)
        ) and not selector.policy_heads:
            return False
        if is_value_head(path) and not selector.value_heads:
            return False
        if is_movesleft_head(path) and not selector.movesleft_heads:
            return False
        return True

    return nnx.map_state(mask_fn, params)
