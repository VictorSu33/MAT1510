import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Load Gemma 3 1B IT
# -----------------------------
model_name = "google/gemma-3-1b-it"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

print("Using device:", device, "dtype:", dtype)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
head_dim = hidden_size // num_heads

# -----------------------------
# 2. Helpers
# -----------------------------
def seq_ones(k: int) -> str:
    """Sequence of k ones separated by spaces: '1 1 1 ... 1'."""
    return " ".join(["1"] * k)

def seq_ones_repeated_last(k: int) -> str:
    """Same as seq_ones(k) but last '1' is repeated once."""
    base = ["1"] * k
    base.append("1")
    return " ".join(base)

def final_token_heads(prompt: str) -> torch.Tensor:
    """
    Return last-layer hidden state of the final token, reshaped as (num_heads, head_dim).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # hidden_states: (emb, layer1, ..., layerN)
    last_layer = out.hidden_states[-1]      # (1, seq_len, hidden_size)
    final_token = last_layer[0, -1, :]      # (hidden_size,)
    heads = final_token.view(num_heads, head_dim)
    return heads.to(torch.float32)

def per_head_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L1 distance per head between two (num_heads, head_dim) tensors.
    Returns shape: (num_heads,)
    """
    return torch.mean(torch.abs(a - b), dim=-1)

# -----------------------------
# 3. Prompts for (a) and (c)
# -----------------------------
PROMPT_A = "How many ones are in the following sequences?\n{seq}"
PROMPT_C = "Can you copy the following number?\n{seq}"

tasks = {
    "(a) Counting 1s": PROMPT_A
}

# -----------------------------
# 4. Sequence lengths (few, mostly large k)
# -----------------------------
Ks = [1, 200, 500, 800, 1100, 1500, 2000, 2500, 3000, 3500,4000,4500, 5000, 5500, 6000]
print("Ks:", Ks)

# Storage: for each task, store mean / min / max per k
means = {name: [] for name in tasks}
mins  = {name: [] for name in tasks}
maxs  = {name: [] for name in tasks}

# -----------------------------
# 5. Main experiment loop
# -----------------------------
for task_name, template in tasks.items():
    print(f"\nRunning task: {task_name}")
    for k in Ks:
        base_seq = seq_ones(k)
        rep_seq  = seq_ones_repeated_last(k)

        prompt_base = template.format(seq=base_seq)
        prompt_rep  = template.format(seq=rep_seq)

        heads_base = final_token_heads(prompt_base)   # (num_heads, head_dim)
        heads_rep  = final_token_heads(prompt_rep)    # (num_heads, head_dim)

        d_per_head = per_head_l1(heads_base, heads_rep)  # (num_heads,)
        d_mean = d_per_head.mean().item()
        d_min  = d_per_head.min().item()
        d_max  = d_per_head.max().item()

        means[task_name].append(d_mean)
        mins[task_name].append(d_min)
        maxs[task_name].append(d_max)

        print(f"  k={k:4d}  mean={d_mean:.6f}  min={d_min:.6f}  max={d_max:.6f}")

# -----------------------------
# 6. Plot mean with min/max bands + bf16 epsilon line
# -----------------------------
plt.figure(figsize=(8, 5))

for task_name in tasks:
    m = np.array(means[task_name])
    lo = np.array(mins[task_name])
    hi = np.array(maxs[task_name])

    plt.plot(Ks, m, marker="o", label=task_name)
    plt.fill_between(Ks, lo, hi, alpha=0.2)

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Sequence Length (k)")
plt.ylabel("Per-head L1 distance (final token, last layer)")
plt.title("Per-head representational difference (Gemma-3-1B-IT)")
plt.grid(True, which="both", linestyle="--", alpha=0.3)

# bf16 machine epsilon
bf16_eps = torch.finfo(torch.bfloat16).eps  # 0.0078125
plt.axhline(bf16_eps, linestyle="--", linewidth=1,
            label=f"bf16 eps ≈ {bf16_eps:.3e}")

plt.legend()
plt.tight_layout()
plt.show()



