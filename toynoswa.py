import math
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
d_model = 64

# ---------------- Sinusoidal positional encoding ----------------
def sinusoidal_pos_encoding(max_len: int, d_model: int, device: str):
    """
    Returns positional encodings of shape (max_len, d_model)
    """
    position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------- Simple layer norm (no learnable params) ----------------
def layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: (L, d)
    LN over last dim, per token
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


# ---------------- Single-head causal attention as a pure function ----------------
def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Q, K, V: (L, d)
    Returns: H = Attn(Q,K)V of shape (L, d)
    """
    L, d = Q.shape
    assert K.shape == (L, d)
    assert V.shape == (L, d)

    scores = Q @ K.T / math.sqrt(d)  # (L, L)

    # Causal mask: disallow attending to future tokens
    mask = torch.triu(torch.ones(L, L, device=Q.device), diagonal=1)
    scores = scores.masked_fill(mask == 1, float("-inf"))

    attn = F.softmax(scores, dim=-1)  # (L, L)
    H = attn @ V                      # (L, d)
    return H


# ---------------- One trial for a given L ----------------
def run_single_trial(L: int, d_model: int = 64) -> float:
    """
    One trial:
      - Sample Q, K, V ~ N(0, I) of shape (L, d)
      - Build extended Q*, K*, V* of shape (L+1, d) by duplicating last token
      - Add sinusoidal PEs + pre-attention LN to Q and K
      - Compute causal attention separately for length L and L+1
      - Return L1 distance between last-token outputs: ||h_L - h*_{L+1}||_1
    """
    # 1. Sample Q, K, V for base sequence
    Q = torch.randn(L, d_model, device=device, dtype=dtype) / math.sqrt(d_model)
    K = torch.randn(L, d_model, device=device, dtype=dtype) / math.sqrt(d_model)
    V = torch.randn(L, d_model, device=device, dtype=dtype)/ math.sqrt(d_model)

    # 2. Duplicate last token to get length L+1
    Q_ext = torch.cat([Q, Q[-1:].clone()], dim=0)  # (L+1, d)
    K_ext = torch.cat([K, K[-1:].clone()], dim=0)  # (L+1, d)
    V_ext = torch.cat([V, V[-1:].clone()], dim=0)  # (L+1, d)

    # 3. Positional encodings
    pe = sinusoidal_pos_encoding(L + 1, d_model, device=device)  # (L+1, d)

    pe_short = pe[:L]       # (L, d)
    pe_long = pe[:L+1]      # (L+1, d)

    # 4. Pre-attention LN on (Q + PE) and (K + PE)
    Q_in = layer_norm(Q + pe_short)         # (L, d)
    K_in = layer_norm(K + pe_short)         # (L, d)
    V_in = V                                # leave V as is (could also LN if desired)

    Q_ext_in = layer_norm(Q_ext + pe_long)  # (L+1, d)
    K_ext_in = layer_norm(K_ext + pe_long)  # (L+1, d)
    V_ext_in = V_ext

    # 5. Run causal attention separately => two different attention matrices
    with torch.no_grad():
        H = causal_attention(Q_in, K_in, V_in)          # (L, d)
        H_ext = causal_attention(Q_ext_in, K_ext_in, V_ext_in)  # (L+1, d)

        h_L = H[-1]          # last token of length-L run
        h_Lplus1 = H_ext[-1] # last token of length-(L+1) run

        l2 = torch.norm(h_L - h_Lplus1, p=2).item()

    return l2


# ---------------- Sweep over L with multiple trials ----------------
def run_length_sweep(
    L_values = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    num_trials: int = 100,
    d_model: int = 64,
):
    results = {}
    for L in L_values:
        dists = []
        for _ in range(num_trials):
            d = run_single_trial(L, d_model=d_model)
            dists.append(d)
        dists_tensor = torch.tensor(dists, dtype=torch.float32)
        mean = dists_tensor.mean().item()
        std = dists_tensor.std(unbiased=True).item()
        results[L] = {
            "distances": dists,
            "mean": mean,
            "std": std,
        }
        print(f"L={L:5d} | mean L2 = {mean:.6f} | std = {std:.6f}")
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    results = run_length_sweep(num_trials=100)
