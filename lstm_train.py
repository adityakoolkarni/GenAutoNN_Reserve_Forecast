"""Defines training for LSTM implementation. Note that this currently
implements BATCH GRADIENT DESCENT and not minibatch SGD as is more common. This
is due to relatively long sequences in the dataset making a small total number
of batches available.
"""

import os
import numpy as np
import json
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim

from lstm_module import (
    rampLSTM,
    rampDataloader,
    plot_seq_forecast,
    plot_losses,
    write_log
    )


def main():
    opts = EasyDict()
    opts.read_path = "data/dataset_v03.csv"
    opts.target = "eia"  # either eia or caiso
    opts.splits = [0.80, 0.10, 0.10]
    opts.embed_dim = 47
    opts.latent_dim = 2**8
    opts.context_len = 24*7
    opts.pred_len = 24
    opts.seq_overlap = 24*4
    opts.dropout_p = 0.10
    opts.n_epochs = 2**11
    opts.n_epochs_per_validation_plot = opts.n_epochs * (2**-5)
    opts.lr = 0.001
    opts.model_no = 7
    opts.write_path = f"lstm_results/model_no_{str(opts.model_no).zfill(2)}"
    print("\nConfiguration:")
    print(json.dumps(opts, indent=4))

    assert not os.path.isdir(opts.write_path), "Model no. dir already exists."
    os.makedirs(opts.write_path)

    dataloader = rampDataloader(opts)
    (train_seqs, validation_seqs, test_seqs) = dataloader.preprocess()
    print("\nShape of datasets:")
    print(f"Train dataset dims: {train_seqs.shape}")
    print(f"Validation dataset dims: {validation_seqs.shape}")
    print(f"Test dataset dims: {test_seqs.shape}")

    lstm = rampLSTM(
        opts.embed_dim,
        opts.latent_dim,
        opts.context_len,
        opts.pred_len,
        opts.dropout_p
        )
    optimizer = optim.Adam(lstm.parameters(), lr=opts.lr)

    fixed_validation_seq_idx = np.random.randint(0, len(validation_seqs))
    fixed_validation_seq = validation_seqs[fixed_validation_seq_idx]

    train_losses = []
    validation_losses = []
    validation_forecasts = {}
    print("\nIn training...")
    for i in range(opts.n_epochs):
        # shuffle the training set
        seq_idxs = torch.randperm(len(train_seqs))
        train_batch = train_seqs[seq_idxs]

        # calc training loss and update the model
        lstm.train()
        optimizer.zero_grad()
        criterion = nn.MSELoss()
        context_seqs = train_batch[:, :opts.context_len, :]
        target_seqs = train_batch[:, opts.context_len:, :]
        last_context_points = context_seqs[:, -1, :]
        hidden, cell = lstm.encode(context_seqs)
        output_seqs = lstm.decode(last_context_points, hidden, cell)
        output_seqs_ramp = output_seqs[:, :, -1]
        target_seqs_ramp = target_seqs[:, :, -1]
        epoch_train_loss = criterion(output_seqs_ramp, target_seqs_ramp)
        train_losses.append(epoch_train_loss)
        epoch_train_loss.backward()
        optimizer.step()

        print(f"\nepoch {i} training MSE loss = {epoch_train_loss}")

        # calc validation loss
        lstm.eval()
        with torch.no_grad():
            context_seqs = validation_seqs[:, :opts.context_len, :]
            target_seqs = validation_seqs[:, opts.context_len:, :]
            last_context_point = context_seqs[:, -1, :]
            hidden, cell = lstm.encode(context_seqs)
            output_seqs = lstm.decode(last_context_point, hidden, cell)
            output_seqs_ramp = output_seqs[:, :, -1]
            target_seqs_ramp = target_seqs[:, :, -1]
            epoch_validation_loss = criterion(
                output_seqs_ramp,
                target_seqs_ramp
                )
            validation_losses.append(epoch_validation_loss)

        print(f"epoch {i} validation MSE loss = {epoch_validation_loss}")

        if i % opts.n_epochs_per_validation_plot == 0 or i == opts.n_epochs - 1:
            (
                context_ramp,
                target_ramp,
                forecast_ramp
            ) = plot_seq_forecast(fixed_validation_seq, opts, lstm, dataloader, i)
            key = f"epoch_{str(i).zfill(4)}"
            validation_forecasts[key] = [
                context_ramp.tolist(),
                target_ramp.tolist(),
                forecast_ramp.tolist()
                ]

    write_log(train_losses, validation_losses, validation_forecasts, opts)
    plot_losses(train_losses, validation_losses, opts.write_path)


if __name__ == '__main__':
    main()
