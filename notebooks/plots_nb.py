import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    sys.path.append(os.getcwd())

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import transformers
    from transformers import AutoModel, BertTokenizerFast
    import matplotlib.pyplot as plt
    from src.masked_modules import MaskedTrainingModule
    from src.masked_modules import MaskedTrainingModule
    import torch.nn.functional as F
    from src.train_script import build_model_config
    from src.autoencoder_modules import AEModule, VAEModule
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    from hydra import initialize_config_dir, compose
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import pickle
    from src.omic_dataclass import OmicDataModule
    import matplotlib
    import omegaconf
    import polars as pl
    import umap.plot
    import umap
    import seaborn as sns

    # MPS
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    matplotlib.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.dpi": 150,
            "figure.figsize": (6.5, 5.5),
        }
    )
    return (
        AEModule,
        ConfusionMatrixDisplay,
        F,
        MLPClassifier,
        OmicDataModule,
        PCA,
        TSNE,
        build_model_config,
        compose,
        confusion_matrix,
        initialize_config_dir,
        np,
        os,
        pickle,
        pl,
        plt,
        sns,
        torch,
        train_test_split,
        umap,
    )


@app.cell
def _(pickle):
    with open("data/cleaned_data_matrix.pkl", "rb") as _f:
        matrix = pickle.load(_f)
    with open("data/comb_matrix_labels.pkl", "rb") as _f:
        _, labels = pickle.load(_f)
    with open("data/data_labels.pkl", "rb") as _f:
        sample_ids = pickle.load(_f)
    return matrix, sample_ids


@app.cell
def _():
    from scipy.stats import zscore

    return (zscore,)


@app.cell
def _(OmicDataModule):
    dm = OmicDataModule(
        file_path="data/cleaned_data_matrix.pkl", use_log_space=True
    )
    dm.setup()
    return (dm,)


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="test_config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(AEModule, build_model_config, get_config, matrix):
    _cfg = get_config("raw_linear")
    _cfg.hidden_dim = 2048
    _cfg.n_hidden_layers = 3
    _model_config = build_model_config(_cfg, inp_dim=matrix.shape[-1])

    model2 = AEModule.load_from_checkpoint(
        checkpoint_path="data/epoch=199-step=200.ckpt",  # was logs/OmicBert/uo0a165h/checkpoints/epoch=199-step=200.ckpt
        model_config=_model_config,
    )
    model2.eval()
    return (model2,)


@app.cell
def _(dm):
    train = dm.train_dataloader()
    test = dm.test_dataloader()
    return


@app.cell
def _(F, dm):
    dataset = dm.train_ds
    mask = ~dataset.nan_mask

    _mean1 = dataset.data.mean(axis=0).unsqueeze(0).repeat(dataset.data.shape[0], 1)
    _mean2 = dataset.data.mean(axis=1).unsqueeze(1).repeat(1, dataset.data.shape[1])

    print(F.mse_loss(_mean1[mask], dataset.data[mask]))
    print(F.mse_loss(_mean2[mask], dataset.data[mask]))
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    MLPClassifier,
    confusion_matrix,
    laterality_list,
    plt,
    sns,
    train_test_split,
):
    def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        normalize=True,
        figsize=(6, 5),
        cmap="Blues",
        save_path=None,
    ):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            annot_fmt = ".2f"
        else:
            annot_fmt = "d"

        # Publication-style settings
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "figure.dpi": 300,
            }
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt=annot_fmt,
            cmap=cmap,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        # ax.set_title("Confusion Matrix")

        plt.tight_layout()

        # Save as vector graphic for papers
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()


    def get_layer_mlp(inp_data, inp_labels):
        for layer_idx in range(len(inp_data[0])):
            _X = []
            for _i in range(len(inp_data)):
                _X.append(inp_data[_i][layer_idx].reshape(-1))
            _X_train, _X_test, _y_train, _y_test = train_test_split(
                _X, inp_labels, random_state=42, stratify=inp_labels
            )
            _clf = MLPClassifier(random_state=42, max_iter=1000).fit(
                _X_train, _y_train
            )
            _clf = MLPClassifier(random_state=42, max_iter=1000).fit(
                _X_train, _y_train
            )
            print(
                "Layer idx = ",
                layer_idx,
                ", Accuracy = ",
                _clf.score(_X_test, _y_test),
            )

            acc = _clf.score(_X_test, _y_test)
            _y_pred = _clf.predict(_X_test)
            cm = confusion_matrix(_y_test, _y_pred)
            print("Confusion Matrix:")
            if len(set(inp_labels)) == 2:
                _file_name = "cancer_clf"
                _label_names = ["Normal", "Tumor"]
            else:
                _file_name = "laterality_clf"
                _label_names = laterality_list
            plot_confusion_matrix(
                _y_test,
                _y_pred,
                _label_names,
                save_path=f"images/{_file_name}_conf_matrix_layer-{layer_idx}.pdf",
                normalize=False,
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=_label_names
            )
            disp.plot()
            # plt.title(f"Layer {layer_idx} Confusion Matrix")
            plt.show()

    return (get_layer_mlp,)


@app.cell
def _(matrix, np, torch, zscore):
    _m = torch.Tensor(matrix)
    train_medians = torch.nanmedian(_m, dim=0, keepdim=True).values
    train_medians = torch.nan_to_num(train_medians, nan=0.0)


    def fill_nan(t, medians):
        is_nan = torch.isnan(t)
        t[is_nan] = medians.expand_as(t)[is_nan]
        return t


    train_m = fill_nan(_m, train_medians)
    train_np = train_m.numpy()
    p1 = np.percentile(train_np, 1, axis=0, keepdims=True)
    p99 = np.percentile(train_np, 99, axis=0, keepdims=True)
    p1_t = torch.from_numpy(p1)
    p99_t = torch.from_numpy(p99)

    train_m = train_m.clamp(min=p1_t, max=p99_t)

    train_np = train_m.numpy()
    train_mean = np.mean(train_np, axis=0, keepdims=True)
    train_std = np.std(train_np, axis=0, keepdims=True)

    train_m = torch.from_numpy(zscore(train_np, axis=0, nan_policy="omit")).float()

    train_m = fill_nan(train_m, train_medians)

    train_m = torch.log1p(train_m.clamp(min=0))
    return (train_m,)


@app.cell
def _(pl):
    metadata_df = pl.read_csv(
        "data/glioblastoma/metadata.csv", infer_schema_length=10000
    )
    return (metadata_df,)


@app.cell
def _(metadata_df):
    metadata_df.head()
    return


@app.cell
def _(metadata_df):
    cancer_types_dict = {}
    for i in metadata_df.iter_rows(named=True):
        cancer_types_dict[i["case_id"]] = (
            i["tumor_laterality"]
            if (i["tumor_laterality"] != "NA")
            else "Non Cancerous"
        )
    return (cancer_types_dict,)


@app.cell
def _(cancer_types_dict, sample_ids):
    cancer_labels = []
    cancer_laterality = []
    laterality_list = list(set(cancer_types_dict.values()))
    for _i in sample_ids:
        if _i in cancer_types_dict:
            cancer_laterality.append(laterality_list.index(cancer_types_dict[_i]))
        else:
            cancer_laterality.append(laterality_list.index("Non Cancerous"))
        if _i.endswith("-T"):
            cancer_labels.append(1)
        elif _i in cancer_types_dict:
            cancer_labels.append(int(cancer_types_dict[_i] != "Non Cancerous"))
        else:
            cancer_labels.append(0)
    return cancer_labels, cancer_laterality, laterality_list


@app.cell
def _(model2, np, torch, train_m):
    test_outputs_lin = []
    for _i in range(len(train_m)):
        _l = []
        model2.to("cpu")
        _inp_embed = model2.model.inp_lin(torch.Tensor(train_m[_i]))
        _l.append(_inp_embed.cpu().detach().numpy())
        for _j in model2.model.encoder_layers:
            _inp_embed = _j(_inp_embed)
            _l.append(_inp_embed.cpu().detach().numpy())
        test_outputs_lin.append(_l)
    test_outputs_lin = np.array(test_outputs_lin)
    return (test_outputs_lin,)


@app.cell
def _(F, train_m):
    _m1 = train_m.mean(axis=1).unsqueeze(1).repeat(1, train_m.shape[-1])
    _m2 = train_m.mean(axis=0).unsqueeze(0).repeat(train_m.shape[0], 1)
    print(F.mse_loss(_m1, train_m))
    print(F.mse_loss(_m2, train_m))
    return


@app.cell
def _(F, dm, model2, plt, train_m):
    _idx = 0
    _inp = train_m.clone()
    _pred = model2(_inp)
    _inp[:, 50455:] = 0
    _pred2 = model2(_inp)
    print("Without masking - ", F.mse_loss(_pred[_idx], dm.test_ds.data[_idx]))
    print("With masking - ", F.mse_loss(_pred2[_idx], dm.test_ds.data[_idx]))
    plt.plot(_pred[_idx].cpu().detach().numpy(), label="Without masking", alpha=0.5)
    plt.plot(_pred2[_idx].cpu().detach().numpy(), label="With masking", alpha=0.5)
    plt.plot(train_m[_idx].cpu().detach().numpy(), label="Original", alpha=0.5)
    plt.legend()
    plt.show()
    return


@app.cell
def _(PCA, TSNE, np, plt, umap):
    def get_pca(inp_data, inp_labels, mean=False, title="", label_map=None):
        pca = PCA(n_components=2)
        if mean:
            components = pca.fit_transform(np.mean(inp_data[:, :, :], axis=2))
        else:
            components = pca.fit_transform(inp_data)
        unique_labels = np.unique(inp_labels)
        cmap = plt.cm.get_cmap(
            "tab10" if len(unique_labels) <= 10 else "gist_ncar"
        )(np.linspace(0, 1, len(unique_labels)))
        for idx, label in enumerate(unique_labels):
            mask = inp_labels == label
            plt.scatter(
                components[mask, 0],
                components[mask, 1],
                c=[cmap[idx]],
                s=55,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=label_map.get(label, str(label)) if label_map else str(label),
            )
        plt.xlabel("PC1", fontsize=14)
        plt.ylabel("PC2", fontsize=14)
        # plt.title("PCA clusters from the embeddings of " + title, fontsize=16, fontweight="bold", pad=14)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"images/pca_clusters.pdf")
        plt.show()


    def get_tsne(inp_data, inp_labels, mean=False, title="", label_map=None):
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        if mean:
            tsne_res = tsne.fit_transform(np.mean(inp_data[:, :, :], axis=2))
        else:
            tsne_res = tsne.fit_transform(inp_data)
        unique_labels = np.unique(inp_labels)
        cmap = plt.cm.get_cmap(
            "tab10" if len(unique_labels) <= 10 else "gist_ncar"
        )(np.linspace(0, 1, len(unique_labels)))
        for idx, label in enumerate(unique_labels):
            mask = inp_labels == label
            plt.scatter(
                tsne_res[mask, 0],
                tsne_res[mask, 1],
                c=[cmap[idx]],
                s=55,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=label_map.get(label, str(label)) if label_map else str(label),
            )
        plt.xlabel("t-SNE 1", fontsize=14)
        plt.ylabel("t-SNE 2", fontsize=14)
        # plt.title("t-SNE clusters from the embeddings of " + title, fontsize=16, fontweight="bold", pad=14)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"images/tsne_clusters.pdf")
        plt.show()


    def get_umap(inp_data, inp_labels, mean=False, title="", label_map=None):
        if mean:
            mapper = umap.UMAP().fit(np.mean(inp_data, axis=1))
        else:
            mapper = umap.UMAP().fit(inp_data, axis=1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        unique_labels = np.unique(inp_labels)
        cmap = plt.cm.get_cmap(
            "tab10" if len(unique_labels) <= 10 else "gist_ncar"
        )(np.linspace(0, 1, len(unique_labels)))
        emb = mapper.embedding_
        for idx, label in enumerate(unique_labels):
            mask = np.array(inp_labels) == label
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=[cmap[idx]],
                s=55,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=label_map.get(label, str(label)) if label_map else str(label),
            )
        ax.set_xlabel("UMAP 1", fontsize=14)
        ax.set_ylabel("UMAP 2", fontsize=14)
        ax.set_title(
            "UMAP clusters from the embeddings of " + title,
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"images/umap_clusters.pdf")
        plt.show()


    def generate_plots(input, labels, mean=False, title="", label_map=None):
        get_pca(input, labels, mean=mean, title=title, label_map=label_map)
        get_tsne(input, labels, title=title, mean=mean, label_map=label_map)
        get_umap(input, labels, title=title, mean=mean, label_map=label_map)

    return (generate_plots,)


@app.cell
def _(cancer_labels, generate_plots, test_outputs_lin):
    binary_label_map = {0: "Non Cancerous", 1: "Cancerous"}
    generate_plots(
        test_outputs_lin[:, 3],
        cancer_labels,
        mean=True,
        title="",
        label_map=binary_label_map,
    )
    return


@app.cell
def _(cancer_laterality, generate_plots, laterality_list, test_outputs_lin):
    label_map = {idx: name for idx, name in enumerate(laterality_list)}
    generate_plots(
        test_outputs_lin[:, 3],
        cancer_laterality,
        mean=False,
        title=" ",
        label_map=label_map,
    )
    return


@app.cell
def _(cancer_labels, cancer_laterality, get_layer_mlp, test_outputs_lin):
    print("Test cancer labels- ")
    get_layer_mlp(test_outputs_lin, cancer_labels)
    print("Test cancer laterality - ")
    get_layer_mlp(test_outputs_lin, cancer_laterality)
    return


if __name__ == "__main__":
    app.run()
