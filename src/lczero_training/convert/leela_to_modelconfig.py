import hlo_pb2
from proto import model_config_pb2, net_pb2


def _defaultactivation_to_activation(
    activation: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    return {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: net_pb2.NetworkFormat.ACTIVATION_MISH,
    }[activation]


def leela_to_modelconfig(
    leela_net: net_pb2.Net,
    weights_dtype: hlo_pb2.XlaShapeProto.Type,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> model_config_pb2.ModelConfig:
    assert weights_dtype == hlo_pb2.XlaShapeProto.F32, (
        "Only float32 weights are supported."
    )
    assert leela_net.format.weights_encoding == net_pb2.Format.LINEAR16
    leela_net_format = leela_net.format.network_format
    model_config = model_config_pb2.ModelConfig()

    model_config.defaults.compute_dtype = compute_dtype
    model_config.defaults.activation = _defaultactivation_to_activation(
        leela_net_format.default_activation
    )
    model_config.defaults.ffn_activation = (
        leela_net_format.ffn_activation or model_config.defaults.activation
    )
    assert (
        leela_net_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    ), "Only dense positional embedding is supported, got {}".format(
        net_pb2.NetworkFormat.InputEmbeddingFormat.Name(
            leela_net_format.input_embedding
        )
    )
    assert leela_net_format.policy == net_pb2.NetworkFormat.POLICY_ATTENTION, (
        "Only attention policy is supported, got {}".format(
            net_pb2.NetworkFormat.PolicyFormat.Name(leela_net_format.policy)
        )
    )
    assert leela_net_format.value == net_pb2.NetworkFormat.VALUE_WDL, (
        "Only WDL value is supported, got {}".format(
            net_pb2.NetworkFormat.ValueFormat.Name(leela_net_format.value)
        )
    )
    assert leela_net_format.moves_left == net_pb2.NetworkFormat.MOVES_LEFT_V1, (
        "Only V1 moves left format is supported, got {}".format(
            net_pb2.NetworkFormat.MovesLeftFormat.Name(
                leela_net_format.moves_left
            )
        )
    )

    def size(x: net_pb2.Weights.Layer) -> int:
        return len(x.params) // 2

    assert (
        leela_net_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    weights = leela_net.weights
    model_config.embedding.dense_size = size(weights.ip_emb_preproc_b) // 64
    model_config.embedding.embedding_size = size(weights.ip_emb_b)
    assert size(weights.ip_mult_gate) > 0
    assert size(weights.ip_add_gate) > 0
    model_config.embedding.dff = size(weights.ip_emb_ffn.dense1_b)

    model_config.encoder.num_blocks = len(weights.encoder)
    assert model_config.encoder.num_blocks > 0
    encoder = weights.encoder[0]
    model_config.encoder.d_model = size(encoder.mha.q_b)
    model_config.encoder.heads = weights.headcount
    model_config.encoder.dff = size(encoder.ffn.dense1_b)

    if weights.HasField("smolgen_w"):
        model_config.encoder.smolgen.activation = (
            leela_net_format.smolgen_activation
            or model_config.defaults.activation
        )
        model_config.encoder.smolgen.hidden_channels = (
            size(encoder.mha.smolgen.compress)
            // model_config.embedding.embedding_size
        )
        model_config.encoder.smolgen.gen_size = (
            size(encoder.mha.smolgen.dense2_b) // weights.headcount
        )
        model_config.encoder.smolgen.hidden_size = size(
            encoder.mha.smolgen.dense1_b
        )

    model_config.policy_head.embedding_size = size(
        weights.policy_heads.ip_pol_b
    )
    model_config.policy_head.d_model = size(
        weights.policy_heads.vanilla.ip2_pol_b
    )

    model_config.value_head.num_channels = size(
        weights.value_heads.winner.ip_val_b
    )

    model_config.movesleft_head.num_channels = size(weights.ip_mov_b)

    return model_config
