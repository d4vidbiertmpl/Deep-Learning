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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # pretrain model
    if pretrain_model:
        model = train(config)
    else:
        assert config.model_type in ('RNN', 'LSTM')

        # Initialize the device which to run the model on
        # device = torch.device(config.device)

        # Initialize params for models
        seq_length = config.input_length
        input_dim = config.input_dim
        num_hidden = config.num_hidden
        num_classes = config.num_classes

        # Initialize the model that we are going to use
        if config.model_type == 'RNN':
            model = VanillaRNN(seq_length, input_dim, num_hidden, num_classes, device)
        else:
            model = LSTM(seq_length, input_dim, num_hidden, num_classes, device)

        model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Start analyzing gradients
    batch_inputs, batch_targets, = next(iter(data_loader))

    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    train_output = model.analyze_hs_gradients(batch_inputs)
    loss = criterion(train_output, batch_targets)

    optimizer.zero_grad()
    loss.backward()
    gradient_norms = []
    time_steps = []

    for i, (t, h) in enumerate(model.h_states[::-1]):
        _grad = h.grad

        l2_norm = _grad.norm(2).item()
        gradient_norms.append(l2_norm)

    fig = plt.figure(figsize=(25, 10), dpi=200)
    fig.suptitle('{}: Accuracies over Palindrome Lengths'.format(m), fontsize=40)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(x=p_lengths, y=means, yerr=stds, fmt='-o', linewidth=4, color="g")
    ax.set_xticks(p_lengths)

    ax.set_xlabel('$Palindrome Length$', fontsize=28)
    ax.set_ylabel('$Accuracy$', fontsize=28)

    plt.savefig("part1/figures/{}_accuracies_over_t.png".format(m))
    plt.show()
