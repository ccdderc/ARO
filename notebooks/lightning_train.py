import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    sys.path.append(os.getcwd())

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from functools import reduce
    from transformers import (
        AutoModel,
        BertTokenizerFast,
        BertConfig,
        BertModel,
        Mamba2Config,
        Mamba2Model,
    )
    from gprofiler import GProfiler
    from data_utils import clean_feature_column, convert_id, combine_dfs, map_metab
    import torch.nn.functional as F
    from src.litmodel import LitModel

    from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler,
        SequentialSampler,
    )
    import matplotlib.pyplot as plt

    # MPS
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return F, LitModel, np, plt, torch, train_test_split


@app.cell
def _(np, torch, train_test_split):
    def get_dataset(fourier=False, batch_size=8):
        with open("data/raw_comb_matrix.npy", "rb") as f:
            matrix_data = np.load(f)

        _m = np.log2(np.nan_to_num(matrix_data, nan=0.0) + 7)
        matrix_data_conv = torch.FloatTensor(_m)
        if not fourier:
            train_data, temp_text, _, _ = train_test_split(
                matrix_data_conv,
                [1] * matrix_data.shape[0],
                random_state=42,
                test_size=0.3,
            )
        else:
            matrix_data_conv_fft = torch.log2(
                torch.abs(torch.fft.rfft(torch.FloatTensor(_m), dim=1))[:, :512] + 1
            )  ## Using log scaler to check improvement in perf.

            train_data, temp_text, _, _ = train_test_split(
                matrix_data_conv_fft,
                [1] * matrix_data.shape[0],
                random_state=42,
                test_size=0.3,
            )

        val_data, test_data, _, _ = train_test_split(
            temp_text, [1] * temp_text.shape[0], random_state=42, test_size=0.5
        )

        train_data = torch.Tensor(train_data)
        val_data = torch.Tensor(val_data)
        test_data = torch.Tensor(test_data)

        # define a batch size
        batch_size = batch_size

        return train_data, val_data, test_data

    return (get_dataset,)


@app.cell
def _(get_dataset):
    train, val, test = get_dataset(batch_size=1)
    return (test,)


@app.cell
def _(F, test, torch):
    _mean_pred = torch.mean(test, axis=1).unsqueeze(1).repeat(1, test.shape[1])
    _median_pred = (
        torch.median(test, axis=1).values.unsqueeze(1).repeat(1, test.shape[1])
    )
    _min = torch.min(test)
    _max = torch.max(test)
    print("Mean - ", F.mse_loss(_mean_pred, test))
    print("Median - ", F.mse_loss(_median_pred, test))
    for _ in range(3):
        _rand_pred = (_max - _min) * torch.rand_like(test) + _min
        print("Random - ", F.mse_loss(_rand_pred, test))
    return


@app.cell
def _():
    return


@app.cell
def _(LitModel):
    model = LitModel.load_from_checkpoint(
        "logs/OmicBert/kw7xefp4/checkpoints/epoch=7-step=640.ckpt"
    ).to("cpu")
    # model = LitModel.load_from_checkpoint('/Users/amogh/Projects/omic_bert/data/epoch=68-step=69.ckpt')
    return (model,)


@app.cell
def _(F, model, plt, test):
    _idx = 3
    _pred = model(test[_idx].reshape(1, -1))
    print("Pred - ", _pred.shape)
    print(test.shape)
    print("MSE - ", F.mse_loss(test[_idx], _pred))
    plt.figure(figsize=(12, 6))
    plt.plot(test[_idx].cpu().detach().numpy(), label="Orig")
    plt.plot(_pred.cpu().detach().numpy(), label="Pred")
    plt.legend()
    plt.savefig("comparison_plot.png", dpi=1200)
    # plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
