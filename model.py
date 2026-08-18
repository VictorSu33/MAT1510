import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()

        self.d_model = d_model
        self.base = base

    def forward(self, seq_len: int, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(
                0,
                self.d_model,
                2,
                device=device,
                dtype=torch.float32,
            )
            * (
                -torch.log(torch.tensor(self.base))
                / self.d_model
            )
        )

        pe = torch.zeros(
            seq_len,
            self.d_model,
            device=device,
            dtype=torch.float32,
        )

        pe[:, 0::2] = torch.sin(
            positions * frequencies
        )

        pe[:, 1::2] = torch.cos(
            positions
            * frequencies[: pe[:, 1::2].shape[1]]
        )

        # Shape: (sequence, d)
        return pe.to(dtype=dtype)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, initialization: str = "orthogonal"):
        super().__init__()

        self.d_model = d_model

        # Parameter-free LayerNorm.
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.reset_parameters(initialization)

    def reset_parameters(self, initialization):
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue

            if initialization == "orthogonal":
                nn.init.orthogonal_(module.weight)

            elif initialization == "xavier":
                nn.init.xavier_uniform_(module.weight)

            # elif initialization == "identity":
            #     if module.weight.shape[0] != module.weight.shape[1]:
            #         raise ValueError(
            #             "Identity initialization requires "
            #             "square matrices."
            #         )
            #     nn.init.eye_(module.weight)

            else:
                raise ValueError(
                    f"Unknown initialization: {initialization}"
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Pre-LN attention block.
        normalized_x = self.norm1(x)

        q = self.q_proj(normalized_x)
        k = self.k_proj(normalized_x)
        v = self.v_proj(normalized_x)

        attention_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x + self.out_proj(attention_output)
        x = x + self.ffn(self.norm2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, depth: int, initialization: str = "orthogonal"):
        super().__init__()

        self.d_model = d_model
        self.depth = depth

        self.positional_encoding = SinusoidalPosEnc(d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, initialization=initialization)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(
            d_model,
            elementwise_affine=False,
        )

    def forward(self, x, apply_final_norm: bool = False):
        """
        x: (batch, sequence, d_model)
        """

        seq_len, d_model = x.shape[-2:]

        assert d_model == self.d_model

        pe = self.positional_encoding(
            seq_len,
            device=x.device,
            dtype=x.dtype,
        )

        hidden_states = x + pe

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # V^(L) final layer representation.
        if not apply_final_norm:
            return hidden_states

        # y = LN_final(V^(L)).
        return self.final_norm(hidden_states)
