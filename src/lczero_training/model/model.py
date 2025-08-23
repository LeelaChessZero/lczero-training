import jax
import jax.numpy as jnp
from flax import nnx

from hlo_pb2 import XlaShapeProto
from proto import net_pb2
from proto.model_config_pb2 import ModelConfig

from .utils import get_activation, get_dtype


class LczeroModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self._input_channels = 112
        self._rngs = rngs

        self.embedding = Embedding(
            input_channels=self._input_channels,
            dense_size=config.embedding_dense_sz,
            embedding_size=config.embedding_size,
            dff=config.encoder_dff,
            default_activation=config.default_activation,
            ffn_activation=config.ffn_activation,
            rngs=self._rngs,
        )

        assert self.config.policy == net_pb2.NetworkFormat.POLICY_ATTENTION
        assert self.config.encoder_layers > 0

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.astype(x, get_dtype(self.config.activations_type))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))

        x = self.embedding(x)
        return x


class Embedding(nnx.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        dense_size: int,
        embedding_size: int,
        dff: int,
        default_activation: net_pb2.NetworkFormat.ActivationFunction,
        ffn_activation: net_pb2.NetworkFormat.ActivationFunction,
        rngs: nnx.Rngs,
    ):
        self._input_channels = input_channels
        self.dense_size = dense_size
        self.default_activation = default_activation

        assert dense_size > 0
        self.preprocess = nnx.Linear(
            in_features=64 * 12,
            out_features=64 * dense_size,
            rngs=rngs,
        )

        assert embedding_size > 0
        self.square_embedding = nnx.Linear(
            in_features=input_channels + dense_size,
            out_features=embedding_size,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embedding_size, rngs=rngs)
        self.ma_gating = MaGating(feature_shape=(embedding_size,), rngs=rngs)
        self.ffn = Ffn(
            in_features=embedding_size,
            layer1_features=dff,
            layer2_features=embedding_size,
            layer1_activation=ffn_activation,
            rngs=rngs,
        )
        self.out_norm = nnx.LayerNorm(embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        pos_info = x[..., :12]
        pos_info = jnp.reshape(pos_info, (64 * 12,))
        pos_info = self.preprocess(pos_info)
        pos_info = jnp.reshape(pos_info, (64, self.dense_size))
        x = jnp.concat((x, pos_info), axis=1)

        # Square embedding.
        x = self.square_embedding(x)
        x = get_activation(self.default_activation)(x)
        x = self.norm(x)
        x = self.ma_gating(x)
        x = self.ffn(x)
        x = self.out_norm(x)
        return x


class Ffn(nnx.Module):
    def __init__(
        self,
        in_features: int,
        layer1_features: int,
        layer2_features: int,
        layer1_activation: net_pb2.NetworkFormat.ActivationFunction,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(
            in_features=in_features, out_features=layer1_features, rngs=rngs
        )
        self.activation = layer1_activation
        self.linear2 = nnx.Linear(
            in_features=layer1_features, out_features=layer2_features, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = get_activation(self.activation)(x)
        x = self.linear2(x)
        return x


class MaGating(nnx.Module):
    def __init__(self, feature_shape: tuple[int, ...], *, rngs: nnx.Rngs):
        self.mult_gate = Gating(
            feature_shape=feature_shape, additive=False, rngs=rngs
        )
        self.add_gate = Gating(
            feature_shape=feature_shape, additive=True, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.mult_gate(x)
        x = self.add_gate(x)
        return x


class Gating(nnx.Module):
    def __init__(
        self,
        feature_shape: tuple[int, ...],
        additive: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.additive = additive
        init_val = 0.0 if self.additive else 1.0
        self.gate = nnx.Param(
            jnp.full(feature_shape, init_val, dtype=jnp.float32)
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.additive:
            return inputs + self.gate.value
        else:
            effective_gate = jax.nn.relu(self.gate.value)
            return inputs * effective_gate


def _tmp_make_config() -> ModelConfig:
    config = ModelConfig()
    config.encoder_layers = 15
    config.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    config.default_activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    config.weights_type = XlaShapeProto.F32
    config.activations_type = XlaShapeProto.BF16
    config.embedding_dense_sz = 512
    config.embedding_size = 1024
    config.encoder_dff = 1536
    return config


if __name__ == "__main__":
    rngs = nnx.Rngs(params=42)
    model = LczeroModel(config=_tmp_make_config(), rngs=rngs)
    # batch_model = nnx.vmap(model, in_axes=0, out_axes=0)
    print(model)

    key = jax.random.key(0)
    random_input = jax.random.normal(key, (112, 8, 8))
    output = model(random_input)
    print(output)
    print(output.shape)
