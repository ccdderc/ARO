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
    from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    import copy
    import marimo as mo

    # MPS
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    matplotlib.style.use("default")
    return (
        AEModule,
        F,
        MLPClassifier,
        OmicDataModule,
        PCA,
        TSNE,
        build_model_config,
        compose,
        confusion_matrix,
        copy,
        initialize_config_dir,
        mo,
        nn,
        np,
        os,
        pickle,
        pl,
        plt,
        sns,
        sys,
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
def _(dm):
    test_ds = dm.test_ds
    return (test_ds,)


@app.cell
def _(test_ds):
    mean_pred = (
        test_ds.data.mean(axis=1).unsqueeze(1).repeat(1, test_ds.data.shape[-1])
    )
    mean_pred_2 = (
        test_ds.data.mean(axis=0).unsqueeze(0).repeat(test_ds.data.shape[0], 1)
    )
    return mean_pred, mean_pred_2


@app.cell
def _(F, mean_pred, mean_pred_2, test_ds):
    print(F.mse_loss(test_ds.data[~test_ds.nan_mask], mean_pred[~test_ds.nan_mask]))
    print(
        F.mse_loss(test_ds.data[~test_ds.nan_mask], mean_pred_2[~test_ds.nan_mask])
    )
    return


@app.cell
def _(F, dm, torch):
    dataset = dm.test_ds
    mask = ~dataset.nan_mask

    _mean1 = dataset.data.mean(axis=0).unsqueeze(0).repeat(dataset.data.shape[0], 1)
    _mean2 = dataset.data.mean(axis=1).unsqueeze(1).repeat(1, dataset.data.shape[1])
    _median1 = (
        dataset.data.median(axis=0)
        .values.unsqueeze(0)
        .repeat(dataset.data.shape[0], 1)
    )
    _median2 = (
        dataset.data.median(axis=1)
        .values.unsqueeze(1)
        .repeat(1, dataset.data.shape[1])
    )

    _rand = torch.rand_like(dataset.data) * dataset.data.std() + dataset.data.mean()

    means = torch.nanmean(dataset.data, dim=0)

    sq_diff = (dataset.data - means) ** 2
    var = torch.nanmean(sq_diff, dim=0)
    stds = torch.sqrt(var)
    means = torch.nan_to_num(means, nan=0.0)
    stds = torch.nan_to_num(stds, nan=0.0)
    _rand_feat = torch.randn_like(dataset.data) * stds + means

    mse_val = F.mse_loss(_rand_feat[mask], dataset.data[mask])

    print(F.mse_loss(_mean1[mask], dataset.data[mask]))
    print(F.mse_loss(_mean2[mask], dataset.data[mask]))
    print(F.mse_loss(_rand[mask], dataset.data[mask]))
    print(f"MSE (Feature-wise Normal): {mse_val.item():.4f}")

    print(F.mse_loss(_median1[mask], dataset.data[mask]))
    print(F.mse_loss(_median2[mask], dataset.data[mask]))
    return dataset, mask


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="test_config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _():
    # import copy

    # dict2 = torch.load(
    #     "logs/epoch=299-step=300.ckpt", map_location=torch.device("cpu")
    # )
    # new_dict = copy.deepcopy(dict2)
    # temp = {}
    # for k in dict2["state_dict"]:
    #     new_k = k.replace("_orig_mod.", "")
    #     temp[new_k] = dict2["state_dict"][k]
    # dict2["state_dict"] = temp
    # torch.save(dict2, "logs/saved_raw_lin.ckpt")
    return


@app.cell
def _(AEModule, build_model_config, get_config, test_ds):
    _cfg = get_config("raw_linear")
    _cfg.hidden_dim = 2048
    _cfg.n_hidden_layers = 3
    _model_config = build_model_config(_cfg, inp_dim=test_ds.data.shape[-1])

    model2 = AEModule.load_from_checkpoint(
        checkpoint_path="logs/OmicBert/uo0a165h/checkpoints/last.ckpt",
        # checkpoint_path="logs/saved_raw_lin.ckpt",
        # checkpoint_path= "data/epoch=199-step=200.ckpt", #"logs/OmicBert/uo0a165h/checkpoints/epoch=199-step=200.ckpt",
        # checkpoint_path="logs/epoch=299-step=300.ckpt",
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
def _(F, dataset, mask, model2):
    model2.to("cpu")
    _preds = model2(dataset.data)
    model2.eval()
    print(F.mse_loss(_preds[mask], dataset.data[mask]))
    print(F.huber_loss(_preds[mask], dataset.data[mask]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA and other Comparison
    """)
    return


@app.cell
def _(F, PCA, dm, torch):
    ## Full data reconstruction

    pca = PCA(n_components=80)
    pca.fit(dm.train_ds.data)


    def test_pca(inp_data, data_mask):
        test_latent = pca.transform(inp_data)
        X_reconstructed = pca.inverse_transform(test_latent)
        print(
            F.mse_loss(
                inp_data[data_mask], torch.Tensor(X_reconstructed)[data_mask]
            )
        )
        print(
            F.huber_loss(
                inp_data[data_mask], torch.Tensor(X_reconstructed)[data_mask]
            )
        )

    return pca, test_pca


@app.cell
def _(dm, test_pca):
    test_pca(dm.train_ds.data, ~dm.train_ds.nan_mask)
    return


@app.cell
def _(dm, test_pca):
    test_pca(dm.val_ds.data, ~dm.val_ds.nan_mask)
    return


@app.cell
def _(dm, test_pca):
    test_pca(dm.test_ds.data, ~dm.test_ds.nan_mask)
    return


app._unparsable_cell(
    r"""
    generate_plots(
        pca.transform(train_m),
        cancer_laterality,
        mean=False,
        title=" Cancer labels",
    )|
    """,
    name="_"
)


@app.cell
def _(cancer_labels, cancer_laterality, get_layer_mlp, pca, torch, train_m):
    latent_feats = pca.transform(train_m)
    get_layer_mlp(torch.Tensor(latent_feats).unsqueeze(1), cancer_laterality)
    get_layer_mlp(torch.Tensor(latent_feats).unsqueeze(1), cancer_labels)
    return


@app.cell
def _():
    # from sklearn.cross_decomposition import PLSRegression

    # X_train_rna = dm.train_ds.data[:, :50455]
    # Y_train_others = dm.train_ds.data

    # X_test_rna = dm.test_ds.data[:, :50455]
    # Y_test_others = dm.test_ds.data

    # pls = PLSRegression(n_components=80)
    # pls.fit(X_train_rna, Y_train_others)

    # Y_pred = pls.predict(X_test_rna)

    # print("PLS MSE:", F.mse_loss(torch.Tensor(Y_pred), Y_test_others))
    return


@app.cell
def _(Y_pred, dm, plt):
    plt.plot(Y_pred[0], alpha=0.5)
    plt.plot(dm.test_ds.data[0, 50455:], alpha=0.5)
    return


@app.cell
def _(X_metabolome, X_proteome, X_rna):
    import mofapy2

    # 1. You must split your unified matrix back into views
    # Based on your Table 4: RNA (~45k-26k), Prot (~10k-12k), Metab (~67-216)
    data_list = [[X_rna], [X_proteome], [X_metabolome]]

    ent = mofapy2.entry_point()
    ent.set_data_matrix(data_list)
    ent.set_model_options(factors=80)  # Match your PCA n_components
    ent.set_train_options(iter=100, convergence_mode="fast")
    ent.build()
    ent.run()

    # Get the latent embeddings (factors)
    weights = ent.get_weights()
    factors = ent.get_factors()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model analysis
    """)
    return


@app.cell
def _(dm, model2, np, pickle, torch):
    from captum.attr import IntegratedGradients

    model2.to("cpu")


    def get_feature_importance(input_tensor, target_idx):
        baseline = torch.zeros_like(input_tensor)
        ig = IntegratedGradients(lambda x: model2(x)[:, target_idx])

        attributions, delta = ig.attribute(
            input_tensor, baseline, return_convergence_delta=True
        )
        return attributions.squeeze().detach().cpu().numpy()


    protein_idx = 57054
    attr_scores = get_feature_importance(
        dm.test_ds.data[0].unsqueeze(0), protein_idx
    )

    top_indices = np.argsort(np.abs(attr_scores))[-100:]
    with open("data/feature_names.pkl", "rb") as _f:
        feat_names = pickle.load(_f)

    print("Target feature - ", feat_names[protein_idx], " Idx - ", protein_idx)
    top_genes = []
    for idx in top_indices:
        top_genes.append(feat_names[idx])
        print(
            f"Feature: {feat_names[idx]}, Importance: {attr_scores[idx]:.4f}, Idx: {idx}"
        )
    return (top_genes,)


@app.cell
def _(top_genes):
    import mygene

    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        top_genes,
        scopes="ensembl.protein,ensembl.gene",
        fields="symbol,name,summary",
        species="human",
    )
    symbols = []
    for res in results:
        symbols.append(res.get("symbol", "N/A"))
        print(f"ID: {res['query']}")
        print(f"Symbol: {res.get('symbol', 'N/A')}")
        print(f"Name: {res.get('name', 'N/A')}\n")
    return (symbols,)


@app.cell
def _(symbols):
    import gseapy as gp

    # Run enrichment against KEGG Pathways
    enr = gp.enrichr(
        gene_list=symbols,
        gene_sets=["KEGG_2021_Human"],
        organism="human",
        outdir=None,
    )

    # Look for pathways with Adjusted P-values < 0.05
    print(enr.results[["Term", "Adjusted P-value"]].head())
    return


@app.cell
def _(dataset, model2, plt):
    model2.to("cpu")
    _preds = model2(dataset.data)
    plt.plot(_preds.detach()[0][:100], label="Pred")
    plt.plot(dataset.data[0][:100], label="Orig")
    plt.legend()
    plt.show()
    return


@app.cell
def _(torch):
    def get_outputs(data, inp_model):
        layer_outputs = []
        for _i in data:
            _embeds = inp_model.model.inp_lin(_i)
            with torch.no_grad():
                _l = []
                for _layer in inp_model.model.encoder_layers:
                    _l.append(_layer(_embeds).cpu().numpy())
                layer_outputs.append(_l)
        return torch.Tensor(layer_outputs)

    return (get_outputs,)


@app.cell
def _(dm, torch):
    comb_inp = torch.cat((dm.train_ds.data, dm.val_ds.data, dm.test_ds.data))
    return (comb_inp,)


@app.cell
def _(matrix, np, torch, zscore):
    matrix_tensor = torch.FloatTensor(matrix)

    train_m = matrix_tensor

    train_nan_mask = torch.isnan(train_m)

    train_medians = torch.nanmedian(train_m, dim=0, keepdim=True).values
    train_medians = torch.nan_to_num(train_medians, nan=0.0)


    def fill_nan(t, medians):
        is_nan = torch.isnan(t)
        t[is_nan] = medians.expand_as(t)[is_nan]
        return t


    train_m = fill_nan(train_m, train_medians)

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

    train_m = fill_nan(train_m, torch.zeros_like(train_medians))

    train_m = torch.log1p(train_m.clamp(min=0))
    return (train_m,)


@app.cell
def _(get_outputs, model2, train_m):
    layer_outputs = get_outputs(train_m, model2)
    return (layer_outputs,)


@app.cell
def _(F, copy, nn, plt, sys, torch):
    class Model(torch.nn.Module):
        def __init__(self, inp_dim, num_classes, hidden_dim=512):
            super().__init__()
            self.inp_dim = inp_dim
            self.num_classes = num_classes
            self.hidden_dim = hidden_dim
            self.lin1 = nn.Linear(inp_dim, self.hidden_dim)
            self.lin2 = nn.Linear(self.hidden_dim, self.num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.lin1(x))
            x = self.dropout(x)
            x = self.lin2(x)
            return x


    class RegrModel(torch.nn.Module):
        def __init__(self, inp_dim, num_classes=1, hidden_dim=512):
            super().__init__()
            self.inp_dim = inp_dim
            self.num_classes = num_classes
            self.hidden_dim = hidden_dim
            self.lin1 = nn.Linear(inp_dim, self.hidden_dim)
            self.lin2 = nn.Linear(self.hidden_dim, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.lin1(x))
            x = self.dropout(x)
            x = self.lin2(x)
            return x


    def train_regr(inp_data, inp_labels, val_data, val_labels):
        model = RegrModel(inp_dim=inp_data.shape[-1])

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )

        val_loss = []
        train_loss = []
        best_weights = None
        lowest_val = sys.maxsize

        for t in range(500):
            model.train()
            optimizer.zero_grad()

            y_pred = model(inp_data)
            loss = loss_fn(y_pred, inp_labels)
            model.eval()
            with torch.no_grad():
                y_pred_val = model(val_data)
                loss_val = loss_fn(y_pred_val, val_labels)
                if loss_val.item() < lowest_val:
                    lowest_val = loss_val.item()
                    best_weights = copy.deepcopy(model.state_dict())

            scheduler.step(loss_val)
            train_loss.append(loss.item())
            val_loss.append(loss_val.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Val")
        plt.legend()
        plt.show()
        model.load_state_dict(best_weights)
        return model


    def train_mlp(inp_data, inp_labels, val_data, val_labels):
        model = Model(
            inp_dim=inp_data.shape[-1], num_classes=len(inp_labels.unique())
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )

        val_loss = []
        train_loss = []
        best_weights = None
        lowest_val = sys.maxsize

        for t in range(500):
            model.train()
            optimizer.zero_grad()

            y_pred = model(inp_data)
            loss = loss_fn(y_pred, inp_labels)
            model.eval()
            with torch.no_grad():
                y_pred_val = model(val_data)
                loss_val = loss_fn(y_pred_val, val_labels)
                if loss_val.item() < lowest_val:
                    lowest_val = loss_val.item()
                    best_weights = copy.deepcopy(model.state_dict())

            scheduler.step(loss_val)
            train_loss.append(loss.item())
            val_loss.append(loss_val.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Val")
        plt.legend()
        plt.show()
        model.load_state_dict(best_weights)
        return model

    return


@app.cell
def _(
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
            print(len(set(inp_labels)))
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
            # disp = ConfusionMatrixDisplay(
            #     confusion_matrix=cm, display_labels=_label_names
            # )
            # disp.plot()
            # #plt.title(f"Layer {layer_idx} Confusion Matrix")
            # plt.show()
    return (get_layer_mlp,)


@app.cell
def _(pickle):
    with open("data/data_labels.pkl", "rb") as _f:
        data_labels = pickle.load(_f)
    return


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
    gender_dict = {}
    bmi_dict = {}
    survival_dict = {}
    for i in metadata_df.iter_rows(named=True):
        cancer_types_dict[i["case_id"]] = (
            i["tumor_laterality"]
            if i["tumor_laterality"] != "NA"
            else "Non Cancerous"
        )
        gender_dict[i["case_id"]] = i["gender"]
        bmi_dict[i["case_id"]] = i["bmi"]
        survival_dict[i["case_id"]] = i["path_diag_to_death_days"]
    return bmi_dict, cancer_types_dict, gender_dict, survival_dict


@app.cell
def _(bmi_dict, cancer_types_dict, gender_dict, sample_ids, survival_dict):
    cancer_labels = []
    cancer_laterality = []
    laterality_list = ["Left", "Right", "Bilateral", "Non Cancerous"]

    gender_labels = []
    bmi_labels = []
    survival_labels = []
    for _i in sample_ids:
        if _i in bmi_dict:
            bmi_labels.append(bmi_dict[_i])
        else:
            bmi_labels.append(-1)

        if _i in gender_dict:
            gender_labels.append(int(gender_dict[_i] == "Male"))
        else:
            gender_labels.append(-1)

        if _i in survival_dict:
            if not survival_dict[_i] == "NA":
                survival_labels.append(int(survival_dict[_i]))
            else:
                survival_labels.append(0)
        else:
            survival_labels.append(-1)

        if _i in cancer_types_dict:
            cancer_laterality.append(laterality_list.index(cancer_types_dict[_i]))
        else:
            cancer_laterality.append(laterality_list.index("Non Cancerous"))
        if _i.endswith("-T"):
            cancer_labels.append(1)
        elif _i in cancer_types_dict:
            cancer_labels.append(int(cancer_types_dict[_i] != "NA"))
        else:
            cancer_labels.append(0)
    return (
        bmi_labels,
        cancer_labels,
        cancer_laterality,
        gender_labels,
        laterality_list,
        survival_labels,
    )


@app.cell
def _(train_m):
    train_m.shape
    return


@app.cell
def _(comb_inp, model2, np, torch, train_m):
    test_outputs_lin = []
    for _i in range(len(comb_inp)):
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
def _(cancer_labels, generate_plots, test_outputs_lin):
    generate_plots(
        test_outputs_lin, cancer_labels, mean=True, title=" Cancer labels"
    )
    return


@app.cell
def _(cancer_labels, generate_plots, layer_outputs):
    generate_plots(
        layer_outputs.detach().cpu().numpy(),
        cancer_labels,
        mean=True,
        title=" Cancer labels",
    )
    return


@app.cell
def _(F, get_layer_mlp, plt, survival_labels, test_outputs_lin, torch):
    _labels = torch.FloatTensor(survival_labels)
    _mask = _labels > 0
    _labels = _labels[_mask]
    _labels = (_labels - _labels.mean()) / (_labels.std() + 1e-8)
    print(F.mse_loss(_labels, _labels.mean().repeat(_labels.shape[0])))
    print(F.mse_loss(_labels, torch.zeros_like(_labels)))
    plt.plot(_labels)
    _inp = torch.Tensor(test_outputs_lin)[_mask]
    get_layer_mlp(_inp, _labels, regr=True)
    return


@app.cell
def _(F, bmi_labels, get_layer_mlp, plt, test_outputs_lin, torch):
    _labels = torch.FloatTensor(bmi_labels)
    _mask = _labels > 0
    _labels = _labels[_mask]
    _labels = (_labels - _labels.mean()) / (_labels.std() + 1e-8)
    print(F.mse_loss(_labels, _labels.mean().repeat(_labels.shape[0])))
    print(F.mse_loss(_labels, torch.zeros_like(_labels)))
    plt.plot(_labels)
    _inp = torch.Tensor(test_outputs_lin)[_mask]
    get_layer_mlp(_inp, _labels, regr=True)
    return


@app.cell
def _(cancer_labels, get_layer_mlp, test_outputs_lin, torch):
    _inp = torch.Tensor(test_outputs_lin)
    get_layer_mlp(_inp, torch.LongTensor(cancer_labels))
    return


@app.cell
def _(cancer_laterality, get_layer_mlp, test_outputs_lin, torch):
    _inp = torch.Tensor(test_outputs_lin)
    get_layer_mlp(_inp, torch.LongTensor(cancer_laterality))
    return


@app.cell
def _(PCA, TSNE, np, plt, umap):
    def get_pca(inp_data, inp_labels, mean=False, title=""):
        pca = PCA(n_components=2)
        if mean:
            components = pca.fit_transform(np.mean(inp_data, axis=1))
        else:
            components = pca.fit_transform(inp_data)
        for label in np.unique(inp_labels):
            idx = inp_labels == label
            plt.scatter(components[idx, 0], components[idx, 1], label=str(label))
        plt.title("PCA clusters from the embeddings of " + title)
        plt.legend()
        plt.show()


    def get_tsne(inp_data, inp_labels, mean=False, title=""):
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        if mean:
            print(np.mean(inp_data, axis=1).shape)
            tsne_res = tsne.fit_transform(np.mean(inp_data, axis=1))
        else:
            tsne_res = tsne.fit_transform(inp_data)
        print(tsne_res.shape)
        plt.title("TSNE clusters from the embeddings of " + title)
        for label in np.unique(inp_labels):
            idx = inp_labels == label
            plt.scatter(tsne_res[idx, 0], tsne_res[idx, 1], label=str(label))
        # plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=inp_labels)
        plt.legend()
        plt.show()


    def get_umap(inp_data, inp_labels, mean=False, title=""):
        if mean:
            mapper = umap.UMAP().fit(np.mean(inp_data, axis=1))
        else:
            mapper = umap.UMAP().fit(inp_data, axis=1)
        umap.plot.points(mapper, labels=np.array(inp_labels))
        plt.title("UMAP clusters from the embeddings of " + title)
        plt.show()


    def generate_plots(input, labels, mean=False, title=""):
        get_pca(input, labels, mean=mean, title=title)
        get_tsne(input, labels, title=title, mean=mean)
        get_umap(input, labels, title=title, mean=mean)

    return generate_plots, get_pca, get_tsne, get_umap


@app.cell
def _(test_outputs_lin):
    test_outputs_lin.shape
    return


@app.cell
def _(gender_labels, generate_plots, np, test_outputs_lin):
    _b = np.array(gender_labels)
    _mask = _b != -1
    _b = _b[_b != -1]
    # _labels = (_b>26).astype(int)
    _labels = _b
    print(_labels)
    generate_plots(test_outputs_lin[_mask], _labels, mean=True, title=" gender")
    return


@app.cell
def _(cancer_laterality, generate_plots, test_outputs_lin):
    generate_plots(
        test_outputs_lin, cancer_laterality, mean=True, title=" Cancer laterality"
    )
    return


@app.cell
def _(
    cancer_labels,
    cancer_laterality,
    get_layer_mlp,
    test_outputs_lin,
    torch,
):
    print("Test cancer labels- ")
    get_layer_mlp(torch.Tensor(test_outputs_lin), torch.LongTensor(cancer_labels))
    print("Test cancer laterality - ")
    get_layer_mlp(
        torch.Tensor(test_outputs_lin), torch.LongTensor(cancer_laterality)
    )
    return


@app.cell
def _(cancer_labels, get_pca, test_outputs_lin):
    get_pca(
        test_outputs_lin,
        cancer_labels,
        mean=True,
        title=" OmicBertTeam/OmicBert/Raw_Linear-best-checkpoint:v15",
    )
    return


@app.cell
def _(
    get_pca,
    test_outputs_label,
    vae_test_outputs_dist,
    vae_test_outputs_lin,
    vae_test_outputs_std,
):
    get_pca(
        vae_test_outputs_lin[:, 0],
        test_outputs_label,
        title="mean from Linear VAE model - j6wa5d08",
    )
    get_pca(
        vae_test_outputs_std[:, 0],
        test_outputs_label,
        title="log var from Linear VAE model - j6wa5d08",
    )
    get_pca(
        vae_test_outputs_dist[:, 0],
        test_outputs_label,
        title="dist from Linear VAE model - j6wa5d08",
    )
    return


@app.cell
def _(
    get_tsne,
    test_outputs_label,
    vae_test_outputs_dist,
    vae_test_outputs_lin,
    vae_test_outputs_std,
):
    get_tsne(
        vae_test_outputs_lin[:, 0],
        test_outputs_label,
        title="mean from Linear VAE model - j6wa5d08",
    )
    get_tsne(
        vae_test_outputs_std[:, 0],
        test_outputs_label,
        title="log var from Linear VAE model - j6wa5d08",
    )
    get_tsne(
        vae_test_outputs_dist[:, 0],
        test_outputs_label,
        title="dist from Linear VAE model - j6wa5d08",
    )
    return


@app.cell
def _(cancer_labels, get_tsne, test_outputs_lin):
    get_tsne(
        test_outputs_lin,
        cancer_labels,
        title=" OmicBertTeam/OmicBert/Raw_Linear-best-checkpoint:v15",
        mean=True,
    )
    return


@app.cell
def _(
    get_umap,
    test_outputs_label,
    vae_test_outputs_dist,
    vae_test_outputs_lin,
    vae_test_outputs_std,
):
    get_umap(
        vae_test_outputs_lin[:, 0],
        test_outputs_label,
        title="mean from Linear VAE model - j6wa5d08",
    )
    get_umap(
        vae_test_outputs_std[:, 0],
        test_outputs_label,
        title="log var from Linear VAE model - j6wa5d08",
    )
    get_umap(
        vae_test_outputs_dist[:, 0],
        test_outputs_label,
        title="dist from Linear VAE model - j6wa5d08",
    )
    return


@app.cell
def _(cancer_labels, get_umap, test_outputs_lin):
    get_umap(
        test_outputs_lin,
        cancer_labels,
        title="OmicBertTeam/OmicBert/Raw_Linear-best-checkpoint:v15",
        mean=True,
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
