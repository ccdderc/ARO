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
    from src.models import BERT_Arch, BERT_Arch_Fourier, Mamba_Fourier
    from gprofiler import GProfiler
    from data_utils import clean_feature_column, convert_id, combine_dfs, map_metab
    import torch.nn.functional as F
    from src.litmodel_linear import LinearLitModel

    # MPS
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return (
        F,
        LinearLitModel,
        ModelCheckpoint,
        WandbLogger,
        device,
        np,
        os,
        pl,
        torch,
        train_test_split,
    )


@app.cell
def _(WandbLogger, os):
    initial_lr = 0.01
    epochs = 200
    weight_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "logs", "default"
    )
    logger = WandbLogger(
        save_dir="logs",
        name="default",
        project="OmicBert",
        reinit=True,
        entity="as3600-university-of-cambridge",
    )
    return epochs, logger, weight_dir


@app.cell
def _(np):
    with open("data/raw_comb_matrix.npy", "rb") as f:
        matrix_data = np.load(f)
    return (matrix_data,)


@app.cell
def _(matrix_data, np, torch, train_test_split):
    _m = np.log2(np.nan_to_num(matrix_data, nan=0.0) + 7)
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        _m,
        [1] * matrix_data.shape[0],
        random_state=42,  # ;)
        test_size=0.3,
    )

    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=42, test_size=0.5, stratify=temp_labels
    )

    train_text = torch.Tensor(train_text)
    val_text = torch.Tensor(val_text)
    test_text = torch.Tensor(test_text)
    return test_text, train_text


app._unparsable_cell(
    r"""
    from torch.utils.data import (
        TensorDataset
        DataLoader,
        RandomSampler,
        SequentialSampler,
    )

    # define a batch size
    batch_size = 8

    # wrap tensors
    train_data = TensorDataset(train_text)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, num_workers=10
    )

    val_data = TensorDataset(val_text)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=batch_size, num_workers=10
    )

    test_data = TensorDataset(test_text)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size, num_workers=10
    )
    """,
    name="_"
)


@app.cell
def _(train_dataloader):
    next(iter(train_dataloader))[0].shape
    return


@app.cell
def _(train_text):
    train_text.shape
    return


@app.cell
def _(LinearLitModel, device, train_text):
    model = LinearLitModel(
        model_name="Linear_Full",
        initial_lr=0.01,
        weight_decay=0,
        inp_dim=train_text.shape[-1],
        out_dim=train_text.shape[-1],
        hidden_dim=2048,
        n_layers=3,
    )
    model = model.to(device)
    return (model,)


@app.cell
def _(ModelCheckpoint, epochs, logger, pl, weight_dir):
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="val_loss",
            )
        ],
        default_root_dir=weight_dir,
        log_every_n_steps=1,
        accelerator="mps",
        logger=logger,
    )
    return (trainer,)


@app.cell
def _(model, train_dataloader, trainer, val_dataloader):
    trainer.fit(model, train_dataloader, val_dataloader)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(F, model, test_text):
    F.mse_loss(model(test_text), test_text)
    return


@app.cell
def _(model, plt, test_text):
    _idx = 1
    pred = model(test_text[_idx].reshape(1, -1))

    plt.plot(pred.detach().numpy(), label="Pred")
    plt.plot(test_text[_idx], label="Original")
    plt.legend()
    return


@app.cell
def _(test_dataloader, trainer):
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)
    return


@app.cell
def _(data_loader, logger, trainer):
    trainer.validate(ckpt_path="best", dataloaders=data_loader)
    trainer.test(ckpt_path="best", dataloaders=data_loader)
    logger.experiment.finish()
    return


if __name__ == "__main__":
    app.run()
