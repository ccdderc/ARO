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
    import torch.nn.functional as F
    from src.omic_dataclass import OmicDataModule
    from hydra import initialize_config_dir, compose

    # MPS
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return (
        BertConfig,
        BertModel,
        BertTokenizerFast,
        F,
        OmicDataModule,
        compose,
        device,
        initialize_config_dir,
        nn,
        np,
        os,
        torch,
        train_test_split,
    )


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="test_config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(OmicDataModule, get_config):
    cfg = get_config("fourier_linear")
    cfg.hidden_dim = 2048
    dm = OmicDataModule(
            file_path="data/comb_matrix_labels.pkl",
            fourier=cfg.get("fourier", False),
            batch_size=cfg.get("batch_size", 8)
        )
    dm.setup()
    return


@app.cell
def _():
    return


@app.cell
def _(BertConfig, BertModel, BertTokenizerFast):
    num_bins = 40
    bert_config = BertConfig()
    # get bert
    bert = BertModel(bert_config)
    # Load the BERT tokenizer
    _tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return (bert,)


@app.cell
def _(np):
    with open("data/comb_matrix.npy", "rb") as f:
        matrix_data = np.load(f)
    return (matrix_data,)


@app.cell
def _(matrix_data, np, torch, train_test_split):
    ## TODO: Testing with both datasets combined together
    _m = np.nan_to_num(matrix_data, nan=0.0)
    matrix_data_conv = torch.log2(
        torch.abs(torch.fft.rfft(torch.FloatTensor(_m), dim=1))[:, :512] + 1
    )  ## Using log scaler to check improvement in perf.
    # split train dataset into train, validation and test sets
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        matrix_data_conv,
        [1] * matrix_data.shape[0],
        random_state=42,  # ;)
        test_size=0.3,
    )

    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=42, test_size=0.5, stratify=temp_labels
    )

    tokens_train = train_text
    tokens_val = val_text
    tokens_test = test_text

    max_len = 25
    return (
        test_labels,
        tokens_test,
        tokens_train,
        tokens_val,
        train_labels,
        val_labels,
    )


@app.cell
def _(
    test_labels,
    tokens_test,
    tokens_train,
    tokens_val,
    torch,
    train_labels,
    val_labels,
):
    train_seq = torch.tensor(tokens_train)
    train_mask = torch.ones_like(train_seq)
    train_y = torch.tensor(train_labels)

    val_seq = torch.tensor(tokens_val)
    val_mask = torch.ones_like(val_seq)
    val_y = torch.tensor(val_labels)

    test_seq = torch.tensor(tokens_test)
    test_mask = torch.ones_like(test_seq)
    test_y = torch.tensor(test_labels)
    return train_mask, train_seq, train_y, val_mask, val_seq, val_y


@app.cell
def _(torch):
    def apply_random_mask(batch_seq, mask_prob=0.15):
        """
        batch_seq: (batch_size, seq_len)
        Returns: masked_seq, labels, masked_indices
        """
        labels = batch_seq.clone()
        masked_seq = batch_seq.clone()

        probability_matrix = torch.full(masked_seq.shape, mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_seq[masked_indices] = 0.0

        return masked_seq, labels, masked_indices

    return (apply_random_mask,)


@app.cell
def _(train_mask, train_seq, train_y, val_mask, val_seq, val_y):
    from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler,
        SequentialSampler,
    )

    # define a batch size
    batch_size = 1

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return train_dataloader, val_dataloader


@app.cell
def _(bert):
    # (don't) freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = True
    return


@app.cell
def _(BERT_Arch_Fourier, bert, device):
    # pass the pre-trained BERT to our define architecture
    # model = BERT_Arch(bert)
    model = BERT_Arch_Fourier(bert)
    # push the model to GPU
    model = model.to(device)
    return (model,)


@app.cell
def _(model):
    # optimizer from hugging face transformers
    from torch.optim import AdamW

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    return (optimizer,)


@app.cell
def _(np, train_labels):
    from sklearn.utils.class_weight import compute_class_weight

    # compute the class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    print("Class Weights:", class_weights)
    return (class_weights,)


@app.cell
def _(class_weights, device, nn, torch):
    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 100
    return (epochs,)


@app.cell
def _(F, apply_random_mask, device, model, optimizer, torch, train_dataloader):
    # function to train the model
    def train():
        model.train()
        (total_loss, total_accuracy) = (0, 0)
        total_preds = []
        for step, batch in enumerate(train_dataloader):
            batch = [
                r.to(device) for r in batch
            ]  # empty list to save model predictions
            (sent_id, attention_mask, _) = batch
            (masked_input, labels, masked_indices) = apply_random_mask(sent_id)
            model.zero_grad()  # iterate over batches
            preds = model(masked_input, attention_mask)
            masked_preds = preds[masked_indices]
            masked_labels = labels[masked_indices]  # Push to GPU
            loss = F.mse_loss(masked_preds, masked_labels)
            total_loss = total_loss + loss.item()  # We ignore original labels
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # 1. Apply Random Masking
            optimizer.step()  # sent_id is your FFT sequence
        total_preds.append(preds)
        avg_loss = total_loss / len(train_dataloader)
        total_preds = torch.concatenate(total_preds, axis=0)
        return (
            avg_loss,
            total_preds,
        )  # 2. Forward pass  # 3. Compute Loss ONLY on masked tokens  # We filter the preds and labels using the boolean mask  # Use MSE for continuous FFT data  # append the model predictions  # compute the training loss of the epoch  # predictions are in the form of (no. of batches, size of batch, no. of classes).  # reshape the predictions in form of (number of samples, no. of classes)  #returns the loss and predictions

    return (train,)


@app.cell
def _(F, apply_random_mask, device, model, torch, val_dataloader):
    import time

    # function for evaluating the model
    def evaluate():
        print("\nEvaluating...")
        t0 = time.time()
        model.eval()
        (total_loss, total_accuracy) = (0, 0)  # deactivate dropout layers
        total_preds = []
        for step, batch in enumerate(val_dataloader):
            if step % 50 == 0 and (not step == 0):
                elapsed = time.time() - t0
                print(
                    "  Batch {:>5,}  of  {:>5,}.".format(step, len(val_dataloader))
                )  # empty list to save the model predictions
            batch = [t.to(device) for t in batch]
            (sent_id, attention_mask, labels) = batch
            with torch.no_grad():  # iterate over batches
                (masked_input, labels, masked_indices) = apply_random_mask(sent_id)
                preds = model(masked_input, attention_mask)
                masked_preds = preds[
                    masked_indices
                ]  # Progress update every 50 batches.
                masked_labels = labels[masked_indices]
                loss = F.mse_loss(masked_preds, masked_labels)
                total_loss = (
                    total_loss + loss.item()
                )  # Calculate elapsed time in minutes.
            total_preds.append(preds)
        avg_loss = total_loss / len(val_dataloader)
        total_preds = torch.concatenate(total_preds, axis=0)  # Report progress.
        return (
            avg_loss,
            total_preds,
        )  # push the batch to gpu  # deactivate autograd  # compute the validation loss of the epoch  # reshape the predictions in form of (number of samples, no. of classes)

    return (evaluate,)


@app.cell
def _(device):
    device
    return


@app.cell
def _(epochs, evaluate, model, torch, train):
    # set initial loss to infinite
    _best_valid_loss = float("inf")
    _train_losses = []
    # empty lists to store training and validation loss of each epoch
    _valid_losses = []
    for _epoch in range(1):
        print("\n Epoch {:} / {:}".format(_epoch + 1, epochs))
        # for each epoch
        (_train_loss, _) = train()
        (_valid_loss, _) = evaluate()
        if _valid_loss < _best_valid_loss:
            _best_valid_loss = _valid_loss
            torch.save(model.state_dict(), "saved_weights.pt")  # train model
        _train_losses.append(_train_loss)
        _valid_losses.append(_valid_loss)
        print(f"\nTraining Loss: {_train_loss:.3f}")  # evaluate model
        print(
            f"Validation Loss: {_valid_loss:.3f}"
        )  # save the best model  # append training and validation loss
    return


@app.cell
def _(model, torch):
    # load weights of best model
    path = "saved_weights.pt"
    model.load_state_dict(torch.load(path))
    return


@app.cell
def _():
    from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    mamba = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    return (mamba,)


@app.cell
def _(nn, torch):
    class Mamba_Fourier_2(nn.Module):
        def __init__(self, backbone):
            super(Mamba_Fourier_2, self).__init__()

            self.backbone = backbone

            # dropout layer
            self.dropout = nn.Dropout(0.1)

            # relu activation function
            self.relu = nn.ReLU()

            self.fft_projection = nn.Linear(1, 768)

            # dense layer 1
            self.fc1 = nn.Linear(768, 1)

            # dense layer 2 (Output layer)
            # self.fc2 = nn.Linear(512,2)

            # softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)

        # define the forward pass
        def forward(self, sent_id, mask):

            # pass the inputs to the model
            x = sent_id.to(torch.float32)
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            elif len(x.shape) == 1:
                x = x.unsqueeze(0).unsqueeze(-1)

            x = self.fft_projection(x)

            x = self.backbone.backbone(inputs_embeds=x).last_hidden_state

            x = self.fc1(x)

            return x.squeeze(-1)

    return (Mamba_Fourier_2,)


@app.cell
def _(Mamba_Fourier_2, device, mamba):
    model_1 = Mamba_Fourier_2(mamba)
    model_1.to(device)
    return (model_1,)


@app.cell
def _(epochs, evaluate, model_1, torch, train):
    _best_valid_loss = float("inf")
    _train_losses = []
    _valid_losses = []
    for _epoch in range(1):
        print("\n Epoch {:} / {:}".format(_epoch + 1, epochs))
        (_train_loss, _) = train()
        (_valid_loss, _) = evaluate()
        if _valid_loss < _best_valid_loss:
            _best_valid_loss = _valid_loss
            torch.save(model_1.state_dict(), "saved_weights.pt")
        _train_losses.append(_train_loss)
        _valid_losses.append(_valid_loss)
        print(f"\nTraining Loss: {_train_loss:.3f}")
        print(f"Validation Loss: {_valid_loss:.3f}")
    return


if __name__ == "__main__":
    app.run()
