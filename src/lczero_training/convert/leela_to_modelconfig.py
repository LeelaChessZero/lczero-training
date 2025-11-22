from proto import hlo_pb2, model_config_pb2, net_pb2


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

    if weights.policy_heads.HasField("ip_pol_w"):
        model_config.shared_policy_heads_embedding_size = size(
            weights.policy_heads.ip_pol_b
        )

    for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
        if weights.policy_heads.HasField(head_name):
            head = getattr(weights.policy_heads, head_name)
            assert size(head.ip2_pol_b) > 0
            assert not head.HasField("ip_pol_w")
            policy_head = model_config.policy_head.add()
            policy_head.name = head_name
            if not model_config.HasField("shared_policy_heads_embedding_size"):
                policy_head.embedding_size = size(head.ip_pol_b)
            policy_head.d_model = size(head.ip2_pol_b)

    for head_name in ["winner", "q", "st"]:
        if weights.value_heads.HasField(head_name):
            head = getattr(weights.value_heads, head_name)
            assert size(head.ip_val_b) > 0
            value_head = model_config.value_head.add()
            value_head.name = head_name
            value_head.num_channels = size(head.ip_val_b)

    movesleft_head = model_config.movesleft_head.add()
    movesleft_head.name = "main"
    movesleft_head.num_channels = size(weights.ip_mov_b)

    return model_config
