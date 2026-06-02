"""Compare model predictions against actual values as overlaid bar charts.

Works with both Linear and Attention model checkpoints.

Usage:
    python -m src.output_analysis \
        --checkpoint path/to/checkpoint.ckpt \
        --indices 0 5 10 \
        --fourier \
        --output-dir output_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.litmodel import MaskedTrainingModule
from src.models import AttentionModel
from src.omic_dataclass import OmicDataModule


def plot_prediction_comparison(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_idx: int,
    output_dir: Path,
) -> None:
    """Plot bar charts of actual vs predicted values."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Output comparison — sample {sample_idx}", fontsize=14)

    x = np.arange(len(actual))

    sns.barplot(x=x, y=actual, color="steelblue", ax=axes[0])
    axes[0].set(xticklabels=[], xlabel="", ylabel="Actual")

    sns.barplot(x=x, y=predicted, color="coral", ax=axes[1])
    axes[1].set(xticklabels=[], xlabel="", ylabel="Predicted")

    sns.barplot(x=x, y=actual, color="steelblue", alpha=0.5, label="Actual", ax=axes[2])
    sns.barplot(
        x=x, y=predicted, color="coral", alpha=0.5, label="Predicted", ax=axes[2]
    )
    axes[2].set(xticklabels=[], xlabel="Feature index", ylabel="Overlay")
    axes[2].legend()

    fig.tight_layout()
    out_path = output_dir / f"output_sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against actual values."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the Lightning .ckpt file",
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
        default="output_analysis",
        help="Directory to save comparison plots",
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
    model = lit_model.model
    use_bf16 = isinstance(model, AttentionModel)

    if use_bf16:
        model = model.to(torch.bfloat16)

    # Load data via OmicDataModule
    dm = OmicDataModule(file_path=args.data, fourier=args.fourier)
    dm.setup()
    # Access the full pre-split tensor from train + val + test
    all_data = torch.cat([dm.train_ds.data, dm.val_ds.data, dm.test_ds.data], dim=0).to(
        device
    )

    with torch.no_grad():
        for idx in args.indices:
            if idx >= len(all_data):
                print(
                    f"Skipping index {idx}: out of range (dataset has {len(all_data)} samples)"
                )
                continue

            x = all_data[idx].unsqueeze(0)
            if use_bf16:
                x = x.to(torch.bfloat16)
            pred = model(x).squeeze(0)

            actual = all_data[idx].cpu().float().numpy()
            predicted = pred.cpu().float().numpy()
            plot_prediction_comparison(actual, predicted, idx, output_dir)


if __name__ == "__main__":
    main()
