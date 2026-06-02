"""Compute gradient-based saliency maps per attention layer from a trained
AttentionModel checkpoint.

Usage:
    python -m src.saliency_analysis \
        --checkpoint path/to/checkpoint.ckpt \
        --indices 0 5 10 \
        --fourier \
        --output-dir analysis_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import torch

from src.litmodel import MaskedTrainingModule


def compute_layer_saliency(
    model: torch.nn.Module, x: torch.Tensor
) -> list[torch.Tensor]:
    """Compute gradient-based saliency maps w.r.t. each attention layer's output.

    For each layer, registers a hook on the layer output, runs a forward pass,
    backpropagates from the summed model output, and records |d(output)/d(layer_out)|
    averaged over the hidden dimension — yielding a (S,) saliency vector per layer.

    Args:
        model: The AttentionModel (not the LightningModule wrapper).
        x: Input tensor of shape (1, S) on GPU.

    Returns:
        List of (S,) tensors, one per attention layer.
    """
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = []

    for i, layer in enumerate(model.layers):

        def make_hook(idx):
            def hook_fn(module, input, output):
                output.retain_grad()
                layer_outputs[idx] = output

            return hook_fn

        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Forward pass with gradients enabled
    out = model(x)
    # Scalar target: sum of all outputs
    out.sum().backward()

    for h in hooks:
        h.remove()

    saliency_maps = []
    for i in sorted(layer_outputs.keys()):
        grad = layer_outputs[i].grad  # (1, S, hidden_dim)
        # Absolute gradient magnitude averaged over hidden dim -> (S,)
        saliency = grad.abs().mean(dim=-1).squeeze(0)
        saliency_maps.append(saliency)

    return saliency_maps


def load_data(file_path: str, fourier: bool, device: torch.device) -> torch.Tensor:
    """Load the full dataset and optionally apply the Fourier transform."""
    with open(file_path, "rb") as f:
        matrix_data, _labels = pickle.load(f)

    m = np.nan_to_num(matrix_data, nan=0.0)
    tensor = torch.FloatTensor(m)

    if fourier:
        tensor = torch.log2(torch.abs(torch.fft.rfft(tensor, dim=1))[:, :512] + 1)

    return tensor.to(device)


def plot_saliency_maps(
    saliency_maps: list[torch.Tensor],
    sample_idx: int,
    output_dir: Path,
) -> None:
    """Plot saliency heatmaps for all layers of a single sample.

    One row per layer, showing the (S,) saliency as a 1D heatmap.
    """
    n_layers = len(saliency_maps)

    fig, axes = plt.subplots(
        n_layers,
        1,
        figsize=(12, 2 * n_layers),
        squeeze=False,
    )
    fig.suptitle(f"Gradient saliency — sample {sample_idx}", fontsize=14, y=1.01)

    for layer_idx, saliency in enumerate(saliency_maps):
        ax = axes[layer_idx][0]
        s = saliency.cpu().float().numpy()
        # Display as a 1×S heatmap
        sns.heatmap(
            s.reshape(1, -1),
            ax=ax,
            cmap="inferno",
            xticklabels=False,
            yticklabels=False,
            cbar=True,
        )
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=10)

    fig.tight_layout()
    out_path = output_dir / f"saliency_sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute gradient saliency maps from a trained AttentionModel checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the Lightning .ckpt file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/comb_matrix_labels.pkl",
        help="Path to the dataset pickle file",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        required=True,
        help="Indices into the full dataset to visualize",
    )
    parser.add_argument(
        "--fourier",
        action="store_true",
        help="Apply Fourier transform to the input data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save saliency plots",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint onto GPU
    lit_model = MaskedTrainingModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    lit_model.eval()
    attn_model = lit_model.model.to(torch.bfloat16)

    # Load data onto GPU in bfloat16
    data = load_data(args.data, fourier=args.fourier, device=device).to(torch.bfloat16)

    for idx in args.indices:
        if idx >= len(data):
            print(
                f"Skipping index {idx}: out of range (dataset has {len(data)} samples)"
            )
            continue

        x = data[idx].unsqueeze(0)  # (1, seq_len)
        saliency = compute_layer_saliency(attn_model, x)
        plot_saliency_maps(saliency, idx, output_dir)
        attn_model.zero_grad()


if __name__ == "__main__":
    main()
