import math
from typing import Any, Optional

from flax import nnx

from proto import net_pb2


class LeelaPytreeWeightsVisitor:
    def __init__(self, nnx_state: nnx.State, leela_net: net_pb2.Net) -> None:
        self.leela_net = leela_net
        self.nnx_state = nnx_state

    def run(self) -> None:
        state = self.nnx_state
        weights = self.leela_net.weights
        self.embedding_block(state["embedding"], weights)
        self.encoder_tower(state["encoders"], weights)
        self.policy_head(state["policy_head"], weights.policy_heads)
        self.value_head(state["value_head"], weights.value_heads)
        self.movesleft_head(state["movesleft_head"], weights)

    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(
            nnx_dict["preprocess"],
            weights.ip_emb_preproc_w,
            weights.ip_emb_preproc_b,
        )
        self.matmul(
            nnx_dict["embedding"],
            weights.ip_emb_w,
            weights.ip_emb_b,
        )
        self.layernorm(
            nnx_dict["norm"],
            weights.ip_emb_ln_gammas,
            weights.ip_emb_ln_betas,
        )
        self.tensor(
            nnx_dict["ma_gating"]["mult_gate"]["gate"], weights.ip_mult_gate
        )
        self.tensor(
            nnx_dict["ma_gating"]["add_gate"]["gate"], weights.ip_add_gate
        )
        self.ffn(nnx_dict["ffn"], weights.ip_emb_ffn)
        self.layernorm(
            nnx_dict["out_norm"],
            weights.ip_emb_ffn_ln_gammas,
            weights.ip_emb_ffn_ln_betas,
        )

    def encoder_tower(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        assert weights.HasField("smolgen_w")
        # Shared layer is stored at the point of the first usage.
        self.matmul(
            nnx_dict["encoders"]["layers"][0]["mha"]["smolgen"][
                "weight_gen_dense"
            ],
            weights.smolgen_w,
            None,
        )

        # assert len(nnx_dict["encoders"]["layers"]) == len(weights.encoder)
        for i in range(len(nnx_dict["encoders"]["layers"])):
            self.encoder_block(
                nnx_dict["encoders"]["layers"][i], weights.encoder[i]
            )

    def encoder_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.EncoderLayer
    ) -> None:
        self.mha(nnx_dict["mha"], weights.mha)
        self.layernorm(nnx_dict["ln1"], weights.ln1_gammas, weights.ln1_betas)
        self.ffn(nnx_dict["ffn"], weights.ffn)
        self.layernorm(nnx_dict["ln2"], weights.ln2_gammas, weights.ln2_betas)

    def mha(self, nnx_dict: nnx.State, weights: net_pb2.Weights.MHA) -> None:
        self.matmul(nnx_dict["q"], weights.q_w, weights.q_b)
        self.matmul(nnx_dict["k"], weights.k_w, weights.k_b)
        self.matmul(nnx_dict["v"], weights.v_w, weights.v_b)
        self.smolgen(nnx_dict["smolgen"], weights.smolgen)
        self.matmul(nnx_dict["output_dense"], weights.dense_w, weights.dense_b)

    def smolgen(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.Smolgen
    ) -> None:
        self.matmul(nnx_dict["compress"], weights.compress, None)
        self.matmul(nnx_dict["dense1"], weights.dense1_w, weights.dense1_b)
        self.layernorm(nnx_dict["ln1"], weights.ln1_gammas, weights.ln1_betas)
        self.matmul(nnx_dict["dense2"], weights.dense2_w, weights.dense2_b)
        self.layernorm(nnx_dict["ln2"], weights.ln2_gammas, weights.ln2_betas)

    def layernorm(
        self,
        nnx_dict: nnx.State,
        scales: net_pb2.Weights.Layer,
        biases: net_pb2.Weights.Layer,
    ) -> None:
        self.tensor(nnx_dict["scale"], scales)
        self.tensor(nnx_dict["bias"], biases)

    def policy_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHeads
    ) -> None:
        self.matmul(nnx_dict["tokens"], weights.ip_pol_w, weights.ip_pol_b)
        vanilla = weights.vanilla
        self.matmul(nnx_dict["q"], vanilla.ip2_pol_w, vanilla.ip2_pol_b)
        self.matmul(nnx_dict["k"], vanilla.ip3_pol_w, vanilla.ip3_pol_b)
        self.matmul(nnx_dict["promotion_dense"], vanilla.ip4_pol_w, None)

    def value_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.ValueHeads
    ) -> None:
        winner = weights.winner
        self.matmul(
            nnx_dict["embed"],
            winner.ip_val_w,
            winner.ip_val_b,
        )
        self.matmul(
            nnx_dict["dense1"],
            winner.ip1_val_w,
            winner.ip1_val_b,
        )
        self.matmul(
            nnx_dict["wdl"],
            winner.ip2_val_w,
            winner.ip2_val_b,
        )

    def movesleft_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(nnx_dict["embed"], weights.ip_mov_w, weights.ip_mov_b)
        self.matmul(nnx_dict["dense1"], weights.ip1_mov_w, weights.ip1_mov_b)
        self.matmul(nnx_dict["out"], weights.ip2_mov_w, weights.ip2_mov_b)

    def ffn(self, nnx_dict: nnx.State, ffn: net_pb2.Weights.FFN) -> None:
        self.matmul(nnx_dict["linear1"], ffn.dense1_w, ffn.dense1_b)
        self.matmul(nnx_dict["linear2"], ffn.dense2_w, ffn.dense2_b)

    def matmul(
        self,
        nnx_dict: nnx.State,
        weights: net_pb2.Weights.Layer,
        biases: Optional[net_pb2.Weights.Layer],
    ) -> None:
        self.tensor(nnx_dict["kernel"], weights)
        if biases:
            self.tensor(nnx_dict["bias"], biases)
        else:
            assert "bias" not in nnx_dict

    def tensor(
        self,
        param: Any,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        print(
            param.shape,
            len(leela.params) // 2,
            math.prod(param.shape),
        )
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0
