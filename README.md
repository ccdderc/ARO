# ARO

A multi-omic machine learning framework for training autoencoder models on combined transcriptomics, proteomics, and metabolomics data. The project supports self-supervised masked prediction.

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 recommended)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ARO
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Install optional dependencies:**

   For CUDA/GPU support with FlashAttention:
   ```bash
   uv sync --extra cuda
   ```

   For extended analysis tools:
   ```bash
   uv sync --extra analysis
   ```

   Or install everything:
   ```bash
   uv sync --all-extras
   ```

4. **Set up pre-commit hooks (optional):**
   ```bash
   uv run pre-commit install
   ```

### Training a Model

Run the training script with one of the provided Hydra configs:

```bash
# Train a linear autoencoder
uv run python src/train_script.py --config-name raw_linear

# Train an attention-based model
uv run python src/train_script.py --config-name raw_attention

# Train a VAE
uv run python src/train_script.py --config-name vae_linear
```

Override config values from the command line:

```bash
uv run python src/train_script.py --config-name raw_linear model.hidden_dim=1024 training.lr=1e-4
```

### Running a Hyperparameter Sweep

```bash
uv run wandb sweep sweep.yaml
uv run wandb agent <sweep-id>
```

### Running Notebooks

For Jupyter notebooks:
```bash
uv run jupyter notebook
```

For Marimo apps:
```bash
uv run marimo edit notebooks/lightning_train.py
```

## File Structure

```
ARO/
├── pyproject.toml                  # Project metadata, dependencies, and ruff config
├── uv.lock                         # Locked dependency versions for reproducible installs
├── .python-version                 # Specifies Python 3.12
├── .gitignore                      # Excludes data/, logs/, wandb/, saved weights, etc.
├── .pre-commit-config.yaml         # Pre-commit hooks for ruff linting and formatting
├── metab_mapping.txt               # CSV mapping metabolite IDs to KEGG/HMDB/PubChem identifiers
├── sweep.yaml                      # W&B Bayesian hyperparameter sweep configuration
├── data_utils.py                   # Data loading, ID conversion (g:Profiler), log/FFT transforms, train/val/test splitting
│
├── configs/                        # Hydra training configurations
│   ├── raw_attention.yaml          # Attention model config
│   ├── raw_linear.yaml             # Linear autoencoder config
│   └── vae_linear.yaml             # VAE config
│
├── src/                            # Core library
│   ├── models.py                   # Model definitions: LinearModel, AttentionModel, LinearVAE, FourierFeatureEncoder
│   ├── autoencoder_modules.py      # PyTorch Lightning modules for AE and VAE training
│   ├── masked_modules.py           # BERT-style masked language modeling module
│   ├── omic_dataclass.py           # OmicDataset and OmicDataModule: preprocessing pipeline (imputation, winsorization, z-score normalization)
│   ├── train_script.py             # Main training entrypoint: Hydra config loading, Lightning Trainer setup, W&B logging
│   ├── analysis.py                 # Dimensionality reduction visualizations (PCA, t-SNE, UMAP)
│   ├── output_analysis.py          # Post-training prediction vs. actual bar plot comparisons
│   └── saliency_analysis.py        # Gradient-based layer-wise saliency heatmaps
│
├── notebooks/                      # Experiment scripts (Marimo apps and standalone Python)
│   ├── lightning_train.py          # Marimo app: loads trained BERT model, compares against baselines
│   ├── lightning_train_linear.py   # Linear model training with Hydra config integration
│   ├── plots_nb.py                 # Comprehensive analysis: PCA/t-SNE/UMAP, confusion matrices, survival/BMI prediction, Integrated Gradients
│   ├── test_model.py               # Model checkpoint loading and test set evaluation
│   └── train_notebook.py           # Marimo app: masked prediction training on FFT-transformed omic data
│
├── omics_bert.ipynb                # Jupyter notebook: data preprocessing, combining RNA/protein/metabolomics
├── omics_bert_alt_training.ipynb   # Jupyter notebook: multi-omic BERT with pathway enrichment and downstream classification
├── test_bert.ipynb                 # Jupyter notebook: checkpoint evaluation, confusion matrices, t-SNE/UMAP
└── train_notebook.ipynb            # Jupyter notebook: BERT training on Fourier-transformed data
```
