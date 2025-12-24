import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = self._build(max_len, d_model, device=device, dtype=dtype)
        self.register_buffer("pe", pe, persistent=False)  # (max_len, d_model)

    @staticmethod
    def _build(max_len, d_model, device, dtype):
        position = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, L: int) -> torch.Tensor:
        """
        Return positional encodings for length L.
        Shape: (L, D)
        """
        if L > self.max_len:
            raise ValueError(f"L={L} exceeds max_len={self.max_len}")
        return self.pe[:L, :]


class SimLN(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x, eps: float = 1e-5):
        return (x - self.mean) / (self.std + eps)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, q, k, v):
        L, D = q.shape
        H = self.n_heads
        d_head = self.d_head

        q = q.view(H, L, d_head)
        k = k.view(H, L, d_head)
        v = v.view(H, L, d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # (H, L, L)

        mask = torch.tril(torch.ones(L, L, device=scores.device, dtype=scores.dtype))
        mask = mask.unsqueeze(0).repeat(H, 1, 1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # (H, L, L)
        out = torch.matmul(attn_weights, v)           # (H, L, d_head)

        out = out.transpose(0, 1).contiguous().view(L, D)  # (L, D)

        if L in [1301,1501,1701]:
            pass

        return out


class AttentionLayer(nn.Module):
    """
    q,k,v -> + PE -> attention -> LN-ish -> linear -> LN-ish
    Returns: (h, [pe_q, pe_k, pe_v])
    """
    def __init__(self, d_model=64, n_heads=4, max_len=4096,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.pe = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_len, dtype=dtype, device=device
        )

        self.ln = SimLN(0, math.sqrt(d_model))
        self.attention = SelfAttention(d_model, n_heads=1)  # keeping your original choice
        self.ffn = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v):
        L, D = q.shape
        pos = self.pe(L)  # (L, D)

        # manually add PE
        q_pe = q + pos
        k_pe = k + pos
        v_pe = v + pos

        h = self.ln(self.attention(q_pe, k_pe, v_pe) + v_pe)
        h = self.ln(self.ffn(h) + h)

        return h

def sample_gaussian_sequence(batch, length, d_model, device, dtype):
    x = torch.randn(batch, length, d_model, device=device, dtype=dtype)
    x = x / math.sqrt(d_model)
    return x


def compute_distance(diff, metric: str):
    """
    diff: (1, d_model) tensor
    """
    if metric == "L2":
        return diff.norm(p=2).item()
    elif metric == "L1":
        return diff.abs().sum().item()
    elif metric == "L1_per_dim":
        return diff.abs().mean().item()
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
def run_experiment(
    d_model=64,
    n_heads=1,
    seq_lengths=None,
    num_trials=50,
    dtype=torch.bfloat16,
    device=None,
    distance_metric="L2",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eps = torch.finfo(dtype).eps
    print(f"=== Depth experiment ===")
    print(f"d_model={d_model}, n_heads={n_heads}, dtype={dtype}, eps={eps}")
    print(f"distance_metric={distance_metric}\n")


    mean_dists = []

    with torch.no_grad():
        dists = []
        for i in range(num_trials):
            model = AttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            max_len=max(seq_lengths) + 1,
            dtype=dtype,
            device=device,
            ).to(device=device, dtype=dtype)

            model.eval()
            
            base = sample_gaussian_sequence(3, seq_lengths[0], d_model, device, dtype)
            trial_dists = []
            for j in range(0,len(seq_lengths)):
                if j > 0:
                    delta = seq_lengths[j] - seq_lengths[j-1]
                    base = torch.concat((base, sample_gaussian_sequence(3, delta, d_model, device, dtype)), dim = 1)
                
                rep = torch.cat([base, base[:, -1:, :].clone()], dim=1)

                q,k,v = base

                q_rep, k_rep, v_rep = rep
                h_base = model(q, k, v)
                h_rep= model(q_rep, k_rep, v_rep)

                last_base = h_base[-1, :]
                last_rep = h_rep[-1, :]

                diff = last_base - last_rep
                dist = compute_distance(diff, distance_metric)
                trial_dists.append(dist)


            dists.append(trial_dists)

        
        dists = np.array(dists)
        mean_dists = np.mean(dists,axis=0)
        min_dists = np.min(dists, axis = 0)
        max_dists = np.max(dists, axis = 0)

    results = {
        "lengths": np.array(seq_lengths),
        "mean": np.array(mean_dists),
        "min": min_dists,
        "max": max_dists
    }

    return results, eps

if __name__ == "__main__":
    d_model = 64
    seq_lengths = list(range(1, 2000,100)) # 16..2048
    distance_metric = "L2"                      # "L2", "L1", "L1_per_dim"
    dtype = torch.bfloat16                      # or torch.float32

    run = True

    # ---- Depth experiment ----
    if run:
        results, eps = run_experiment(
            d_model=d_model,
            n_heads=4,          # fixed
            seq_lengths=seq_lengths,
            num_trials= 50,
            dtype=dtype,
            distance_metric=distance_metric,
        )

        plt.figure(figsize=(8, 5))

        Ls = results["lengths"]
        mean = results["mean"]
        min = results["min"]
        max = results["max"]

        print(np.where(mean > eps)[0])
        print(Ls[[13,17]])

        plt.plot(Ls, mean, color = "C0")

        plt.fill_between(Ls, min,max, color="C0", alpha = 0.25)
        
        plt.axhline(
            y=eps,
            linestyle=":",
            linewidth=1.5,
            label=f"machine epsilon ({str(dtype)})",
        )

        plt.yscale("log")
        plt.xlabel("Sequence length L")
        plt.ylabel(f"Mean {distance_metric} distance (last token)")
        plt.title(f"Pure attention: collapse vs depth (d_model={d_model})")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()