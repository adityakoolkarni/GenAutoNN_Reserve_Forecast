"""Simple LSTM implementation to compare to deep state space model forecast
accuracy on EIA and CAISO datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


def write_log(train_losses, validation_losses, validation_forecasts, opts):
    write_path = opts.write_path
    min_train_loss_iter = np.argmin(train_losses)
    min_validation_loss_iter = np.argmin(validation_losses)
    log = {
        "config": opts,
        "results": {
            "min_train_loss_iter": int(min_train_loss_iter),
            "min_train_loss": float(train_losses[min_train_loss_iter]),
            "min_validation_loss_iter": int(min_validation_loss_iter),
            "min_validation_loss": float(validation_losses[min_validation_loss_iter])
        },
        "forecasts": validation_forecasts
    }
    json.dump(log, open(os.path.join(write_path, "log.txt"), "w"), indent=4)


def plot_losses(train_losses, validation_losses, plot_path):
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Training losses")
    ax.plot(validation_losses, label="Validation losses")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    filename = "lstm_losses.png"
    filepath = os.path.join(plot_path, filename)

    plt.savefig(filepath)


def plot_seq_forecast(seq, opts, lstm, dataloader, epoch):
    context_seq = torch.unsqueeze(
        seq[:opts.context_len, :],
        dim=0
        )
    target_seq = torch.unsqueeze(
        seq[opts.context_len:, :],
        dim=0
        )
    last_context_point = context_seq[:, -1, :]
    hidden, cell = lstm.encode(context_seq)
    output = lstm.decode(
        last_context_point,
        hidden,
        cell
        ).detach()
    full_context_ramp = dataloader.inv_scale_ramp(
        context_seq[:, :, -1]
        ).squeeze()
    full_target_ramp = dataloader.inv_scale_ramp(
        target_seq[:, :, -1]
        ).squeeze()
    full_output_ramp = dataloader.inv_scale_ramp(
        output[:, :, -1]
        ).squeeze()

    plot_ramps(full_output_ramp, full_target_ramp, epoch, opts.write_path)
    return full_context_ramp, full_target_ramp, full_output_ramp


def plot_ramps(output, target, epoch, plot_path):
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(output, label="Generated forecast")
    ax.plot(target, label="Target")
    ax.legend()
    ax.set_xlabel('Hour')
    ax.set_ylabel('EIA Net Load Ramp')
    filename = f"epoch_{str(epoch).zfill(4)}.png"
    filepath = os.path.join(plot_path, filename)
    plt.savefig(filepath)


class rampLSTM(nn.Module):
    def __init__(self, embed_dim, latent_dim, context_len, pred_len, p):
        super(rampLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.pred_len = pred_len
        self.dropout = nn.Dropout(p=p)
        self.lstm_cell_encode = nn.LSTMCell(
            embed_dim,
            latent_dim
            )
        self.lstm_cell_decode = nn.LSTMCell(
            embed_dim,
            latent_dim
            )
        self.linear_layer = nn.Linear(
            latent_dim,
            embed_dim
            )
        self.activation = nn.Tanh()

    def encode(self, input_batch_sequences):
        batch_size = input_batch_sequences.shape[0]
        hidden = torch.zeros(batch_size, self.latent_dim)
        cell = torch.zeros(batch_size, self.latent_dim)

        for i in range(self.context_len):
            input_batch = self.dropout(input_batch_sequences[:, i, :])
            hidden, cell = self.lstm_cell_encode(
                input_batch,
                (hidden, cell)
                )

        return hidden, cell

    def decode(self, input_batch_points, init_hidden, init_cell):
        input_batch = input_batch_points
        hidden = init_hidden
        cell = init_cell

        outputs = []
        for i in range(self.pred_len):
            hidden, cell = self.lstm_cell_decode(
                input_batch,
                (hidden, cell)
                )
            output_batch = self.activation(self.linear_layer(hidden))
            outputs.append(output_batch)
            input_batch = output_batch

        outputs = torch.stack(outputs, dim=1)
        return outputs


class rampDataloader():
    def __init__(self, opts):
        self.df = pd.read_csv(opts.read_path).iloc[1:]  # NaN in the first row
        self.target = opts.target
        self.splits = opts.splits
        self.context_len = opts.context_len
        self.pred_len = opts.pred_len
        self.seq_len = self.context_len + self.pred_len
        self.seq_overlap = opts.seq_overlap
        self.scaler = StandardScaler()

    def scale_ramp(self, ramp):
        return self.scaler.fit_transform(ramp)

    def inv_scale_ramp(self, scaled_ramp):
        return self.scaler.inverse_transform(scaled_ramp)

    def preprocess(self):
        oh_enc = OneHotEncoder()
        oh_cats_df = self.df[["cat_hour", "cat_day", "cat_month", "cat_year"]]
        oh_cats = oh_enc.fit_transform(oh_cats_df).toarray()
        if "eia" in self.target:
            target = np.expand_dims(np.asarray(self.df["eia_ramp"]), axis=1)
        elif "caiso" in self.target:
            target = np.expand_dims(np.asarray(self.df["caiso_ramp"]), axis=1)
        scaled_target = self.scale_ramp(target)
        data = np.hstack((oh_cats, scaled_target))
        seqs = self.split_seqs(data)

        train_seqs, validation_seqs, test_seqs = self.split_datasets(seqs)

        return (
            torch.tensor(train_seqs, dtype=torch.float),
            torch.tensor(validation_seqs, dtype=torch.float),
            torch.tensor(test_seqs, dtype=torch.float)
            )

    def split_seqs(self, data):
        step_size = self.seq_len - self.seq_overlap
        seqs = []
        for i in range(0, data.shape[0], step_size):
            seq_end = i + self.seq_len
            if seq_end > data.shape[0]:
                break
            cur_seq = data[i:seq_end]
            seqs.append(cur_seq)

        return np.asarray(seqs, dtype=np.float32)

    def split_datasets(self, seqs):
        partition0 = int(self.splits[0] * len(seqs))
        partition1 = int((self.splits[0] + self.splits[1]) * len(seqs))

        train_seqs = np.asarray(seqs[:partition0])
        validation_seqs = seqs[partition0:partition1]
        test_seqs = seqs[partition1:]

        return [train_seqs, validation_seqs, test_seqs]
