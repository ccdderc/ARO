import pandas as pd
from gprofiler import GProfiler
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import pickle
import torch
from scipy.stats import zscore


def clean_feature_column(df, col, strip_versions=False):
    df[col] = df[col].astype(str)
    df = df[~df[col].str.match(r"\d{4}-\d{2}-\d{2}")].copy()

    ## Adding code to convert ENSG ids and stripping versions
    if strip_versions:
        df[col] = df[col].str.split(".").str[0]
    return df


def convert_id(df, col, replace_col=False, target_namespace="ENSG"):
    gp = GProfiler(return_dataframe=True)
    query_list = df[col].unique().tolist()

    mapping = gp.convert(
        organism="hsapiens", query=query_list, target_namespace=target_namespace
    )
    mapping = mapping[mapping["converted"].notna() & (mapping["converted"] != "None")]
    mapping = mapping[["incoming", "converted"]]
    expanded_df = pd.merge(df, mapping, left_on=col, right_on="incoming", how="inner")
    expanded_df = expanded_df.drop(columns=["incoming"])
    if replace_col:
        expanded_df[col] = expanded_df["converted"]
        expanded_df = expanded_df.drop(columns=["converted"])
    return expanded_df


def get_dataset(fourier=False, batch_size=8, labels=False):
    with open("data/comb_matrix_labels.pkl", "rb") as f:
        matrix_data, labels = pickle.load(f)

    _m = np.log2(np.nan_to_num(matrix_data, nan=0.0) + 7)
    matrix_data_conv = torch.FloatTensor(_m)
    if not fourier:
        train_data, temp_text, train_labels, temp_labels = train_test_split(
            matrix_data_conv,
            labels,
            random_state=42,
            test_size=0.3,
        )
    else:
        matrix_data_conv_fft = torch.log2(
            torch.abs(torch.fft.rfft(torch.FloatTensor(_m), dim=1))[:, :512] + 1
        )  ## Using log scaler to check improvement in perf.

        train_data, temp_text, train_labels, temp_labels = train_test_split(
            matrix_data_conv_fft,
            labels,
            random_state=42,
            test_size=0.3,
        )

    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=42, test_size=0.5
    )

    train_data = torch.Tensor(train_data)
    val_data = torch.Tensor(val_data)
    test_data = torch.Tensor(test_data)

    # define a batch size
    batch_size = batch_size

    # wrap tensors
    # train_data = TensorDataset(train_data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=True,
    )

    # val_data = TensorDataset(val_data)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(
        val_data,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=True,
    )

    # test_data = TensorDataset(test_data)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=True,
    )
    if labels:
        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            train_labels,
            val_labels,
            test_labels,
        )
    return train_dataloader, val_dataloader, test_dataloader


def combine_dfs(dfs):
    cleaned_dfs = []
    for df in dfs:
        df.columns = df.columns.astype(str).str.strip()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        cleaned_dfs.append(df)
    combined = pd.concat(cleaned_dfs, axis=0, sort=False)
    return combined


def map_metab(df, col, map_file_path):
    map_df = pd.read_csv(map_file_path, sep=",")
    mapping_dict = dict(zip(map_df["Query"], map_df["KEGG"]))
    df[col] = df[col].map(mapping_dict)
    df = df.dropna(subset=[col])
    return df


def normalize_data(inp_df):
    data_matrix = inp_df.to_numpy()
    norm_data = zscore(data_matrix, nan_policy="omit")
    inp_df.iloc[:] = norm_data
    return inp_df
