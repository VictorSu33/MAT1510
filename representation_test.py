import gc
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_dists(model, inputs1, inputs2, n, name):
    with torch.inference_mode():
        outputs1 = model(**inputs1, output_hidden_states=True ,use_cache=False)
        outputs2 = model(**inputs2, output_hidden_states=True, use_cache=False)

        # Shape: (num_hidden_states, hidden_size)
        states1 = torch.stack([
            hidden[0, -1] for hidden in outputs1.hidden_states
        ]).float()

        states2 = torch.stack([
            hidden[0, -1] for hidden in outputs2.hidden_states
        ]).float()

        distances = torch.linalg.vector_norm(
            states1 - states2,
            ord=2,
            dim=1,
        ).cpu().numpy()
    
    print("output shape for n =", n, "model", name, ":", outputs1.hidden_states[0].shape)
    return distances


def plot_rep_diff(dists, nums, name):
    num_layers = dists.shape[1]
    x_positions = np.arange(len(nums))

    # First, middle, and last hidden-state levels
    legend_layers = {
        0,
        num_layers // 2,
        num_layers - 1,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Perceptually uniform colour progression
    cmap = plt.colormaps["viridis"]
    norm = plt.Normalize(vmin=0, vmax=num_layers - 1)

    for layer in range(num_layers):
        y = dists[:, layer]

        y = np.where(y > 0, y, np.nan)

        is_legend_layer = layer in legend_layers

        ax.plot(
            x_positions,
            y,
            color=cmap(norm(layer)),
            linewidth=2.5 if is_legend_layer else 1.2,
            alpha=1.0 if is_legend_layer else 0.8,
            marker="o" if is_legend_layer else None,
            markersize=3,
            label=f"Layer {layer}" if is_legend_layer else "_nolegend_",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in nums])

    ax.set_xlabel("n (repeated-sequence length)")
    ax.set_ylabel("L2 distance")
    ax.set_title(f"Representation distance across layers for {name}")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)

    ax.legend(title="Selected layers")

    color_mapping = plt.cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
    )

    colorbar = fig.colorbar(
        color_mapping,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label("Layer")
    colorbar.set_ticks(sorted(legend_layers))
    fig.tight_layout()

    fig.savefig(
        rf"C:\Users\victo\GitHub Projects\MAT1510\images\{name}_all_layer_distance_2d.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "Qwen 2.5": "Qwen/Qwen2.5-1.5B-Instruct",
        "Gemma": "google/gemma-3-1b-it",
        "Llama": "meta-llama/Llama-3.2-1B-Instruct"
    }
    
    # test on repeated sequences

    rep_nums = [1,2,5,10,50,100,500, 1000, 2000]

    for name, model_code in models.items():
        # instantiate model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_code)
        model = AutoModelForCausalLM.from_pretrained(model_code, dtype=torch.float16).to(device)

        model.eval()

        all_dist = []


        for n in rep_nums:
            
            # repeated sequence of 1s length n and n+1 with a token at the start to break homogeneity

            sequence1 = "Please count thew" + "".join(["1"] * n)
            sequence2 = "A" + "".join(["1"] * (n + 1))

            # inputs for calculating representation distance

            inputs1 = tokenizer(
                sequence1,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)
            inputs2 = tokenizer(
                sequence2,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)


            distances = get_dists(model, inputs1, inputs2, n, name)

            all_dist.append(distances)

        # 2d plot of representation distance curves for each layer   
        all_dist = np.asarray(all_dist)

        plot_rep_diff(all_dist, rep_nums, name)

        del model, tokenizer, all_dist
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
