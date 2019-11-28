from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
from part1.train import train

from torch.autograd import grad


def analyze_grads_over_time(config, pretrain_model=False):
    device = torch.device(config.device)
    config.input_length = 150

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_norms = []

    for m in ["RNN", "LSTM"]:

        # pretrain model
        if pretrain_model:
            model = train(config)
        else:
            # Initialize params for models
            seq_length = config.input_length
            input_dim = config.input_dim
            num_hidden = config.num_hidden
            num_classes = config.num_classes

            # Initialize the model that we are going to use
            if m == 'RNN':
                model = VanillaRNN(seq_length, input_dim, num_hidden, num_classes, device)
            else:
                model = LSTM(seq_length, input_dim, num_hidden, num_classes, device)

            model.to(device)

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(config.input_length + 1)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=1)

        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

        # Get single batch from dataloader
        batch_inputs, batch_targets, = next(iter(data_loader))

        # convert to one-hot
        batch_inputs = torch.scatter(torch.zeros(*batch_inputs.size(), num_classes), 2,
                                     batch_inputs[..., None].to(torch.int64), 1).to(device)
        batch_targets = batch_targets.to(device)

        train_output = model.analyze_hs_gradients(batch_inputs)
        loss = criterion(train_output, batch_targets)

        optimizer.zero_grad()
        loss.backward()

        gradient_norms = []
        for i, (t, h) in enumerate(model.h_states[:-1]):
            _grad = h.grad  # (batch_size x hidden_dim)
            average_grads = torch.mean(_grad, dim=0)  # Calculate average gradient to get more stable estimate
            grad_l2_norm = average_grads.norm(2).item()
            gradient_norms.append(grad_l2_norm)

    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.suptitle('L2-norm of Gradients across Time Steps', fontsize=36)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(total_norms[0], linewidth=2, color="tomato", label="RNN")
    ax.plot(total_norms[1], linewidth=2, color="darkblue", label="LSTM")
    ax.tick_params(labelsize=16)

    ax.set_xlabel('Time Step', fontsize=24)
    ax.set_ylabel('Gradient Norm (L2)', fontsize=24)
    ax.legend()

    plt.savefig("part1/figures/Analyze_gradients_pt_{}.png".format(str(pretrain_model)))
    plt.show()
