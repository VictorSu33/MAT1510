import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_pos_encoding(
    max_len: int,
    d_model: int,
    device: str = "cpu",
):
    """
    Returns positional encodings with shape (max_len, d_model).
    """
    position = torch.arange(
        max_len,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    frequency = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
            device=device,
        )
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(
        max_len,
        d_model,
        dtype=torch.float32,
        device=device,
    )

    pe[:, 0::2] = torch.sin(position * frequency)

    pe[:, 1::2] = torch.cos(
        position * frequency[: pe[:, 1::2].shape[1]]
    )

    return pe

def transformer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor, sliding_window_size: int = 0):
    """
    q, k, v: (sequence, d)
    pe: (sequence, d)
    sliding_window_size: int, optional
        If > 0, apply a sliding window mask to the attention scores. If 0, only causal mask is applied.

    Returns: (batch, sequence, d)
    """
    seq_len, d = q.size()
    assert k.size() == (seq_len, d)
    assert v.size() == (seq_len, d)
    assert pe.size() == (seq_len, d)
    assert sliding_window_size >= 0

    pe = pe.to(device=q.device, dtype=q.dtype)
    # q, k, v, positional_encoding: (batch, sequence, d)
    q_pos = q + pe
    k_pos = k + pe

    scores = q_pos @ k_pos.transpose(-2, -1)
    scores = scores / math.sqrt(d)

    positions = torch.arange(seq_len, device=q.device)
    query_positions = positions.unsqueeze(1)
    key_positions = positions.unsqueeze(0)

    # causal mask
    mask = key_positions > query_positions

    # sliding window
    if sliding_window_size > 0:
        # Window size includes the current token.
        outside_window = (
            key_positions
            < query_positions - sliding_window_size + 1
        )
        mask = mask | outside_window

    scores = scores.masked_fill(
            mask.unsqueeze(0),
            float("-inf"),
        )
    attention_weights = torch.softmax(scores, dim=-1)

    # parameter free normalization
    normalized_v = F.layer_norm(
        v,
        normalized_shape=(d,),
        weight=None,
        bias=None,
    )

    # V^(1) = V + A LN(V)
    attention_output = attention_weights @ normalized_v
    v_out = v + attention_output

    # y = LN_final(V^(1))
    y = F.layer_norm(
        v_out,
        normalized_shape=(d,),
        weight=None,
        bias=None,
    )

    return {
    "v_out": v_out,                 # V^(1), theorem quantity
    "y": y,      # final model representation
    "attention": attention_weights,
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_lens = [10, 50, 100, 500, 1000, 2000]
    seeds = [n for n in range(10)]
    dim = 64
    windows = [0, 4, 16, 124]
   
    max_len = max(seq_lens)

    pe = sinusoidal_pos_encoding(
        max_len=max_len + 1,
        d_model=dim,
        device=device,
    )

    distance_array = []
    distance_array_swa = []

    for seed in seeds:
        torch.manual_seed(seed)
        q, k, v = torch.randn(3,max(seq_lens), dim, device=device) / dim**0.5
        seed_distances = []
        for window in windows:

            with torch.inference_mode(): 
                output_dict = transformer(q, k, v, pe[:max_len], sliding_window_size=window)
            v_out = output_dict["v_out"]

            distances = []

            for seq_len in seq_lens:

                # causal 
                print(f"Seed: {seed}, Seq_len: {seq_len}, Output shape: {v_out.shape}")
                v_last = v_out[0, seq_len-1]

                q_n, k_n, v_n = q[:seq_len], k[:seq_len], v[:seq_len]
                q_repeated = torch.cat([q_n, q_n[-1:]], dim=0)
                k_repeated = torch.cat([k_n, k_n[-1:]], dim=0)
                v_repeated = torch.cat([v_n, v_n[-1:]], dim=0)
                pe_repeat = sinusoidal_pos_encoding(seq_len + 1, dim, device)
                output_repeat_dict = transformer(q_repeated, k_repeated, v_repeated, pe_repeat)
                v_last_repeat = output_repeat_dict["v_out"][0, -1]

                distance = torch.linalg.vector_norm(v_last - v_last_repeat, ord=1).cpu().numpy()
                print(f"Distance for Seed {seed}, seq_len {seq_len}: {distance}")


                distances.append(distance)

            seed_distances.append(distances)

        distance_array.append(seed_distances)

    distance_array = np.array(distance_array)
    print(distance_array.shape)
    assert distance_array.shape == (
        len(seeds),
        len(windows),
        len(seq_lens),
    )

    x_positions = np.arange(len(seq_lens))  
    
    mean_distances = np.mean(distance_array, axis=0)
    min_distances = np.min(distance_array, axis=0)
    max_distances = np.max(distance_array, axis=0)

    assert distance_array.shape == (
        len(seeds),
        len(windows),
        len(seq_lens),
    )

    assert np.all(min_distances <= mean_distances)
    assert np.all(mean_distances <= max_distances)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        plt.get_cmap("tab10")(i)
        for i in range(len(windows))
    ]

    markers = ["o", "s", "^", "D"]

    for window_index, window in enumerate(windows):
        lower = min_distances[window_index]
        center = mean_distances[window_index]
        upper = max_distances[window_index]

        color = colors[window_index]

        label = (
            "Full causal attention"
            if window == 0
            else f"Window size {window}"
        )

        ax.fill_between(
            x_positions,
            lower,
            upper,
            color=color,
            alpha=0.15,
            edgecolor="none",
            zorder=1,
        )

        ax.plot(
            x_positions,
            center,
            color=color,
            marker=markers[window_index],
            linewidth=2.2,
            label=label,
            zorder=3,
        )


    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("L1 Distance")
    ax.set_title(f"Representational Collapse in Toy Model")
    ax.legend()
    fig.savefig(
        rf"C:\Users\victo\GitHub Projects\MAT1510\images\toy_model_compare_swa.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)




