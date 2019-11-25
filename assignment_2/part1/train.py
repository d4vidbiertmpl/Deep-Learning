################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

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

import torch.nn.functional as F

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
from torch.utils.tensorboard import SummaryWriter


################################################################################


def local_experiments(config):
    p_lengths = [30, 10, 15, 20, 25, 30, 50]
    learning_rates = [1e-4, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4]
    models = ["LSTM"]

    print(config)

    accuracies_over_t = [[], []]
    for i, m in enumerate(models):
        for j, t in enumerate(p_lengths):
            accuracies = []
            for seed in [42, 43, 44, 45]:
                print("Model: {}, T: {}, Seed: {}".format(m, t, seed))
                torch.manual_seed(seed)
                np.random.seed(seed)
                config.input_length = t
                config.model_type = m
                print("LR", learning_rates[j])
                config.learning_rate = learning_rates[j]

                model = train(config)

                mean_accuracy = evaluate_accuracy(model, config)
                accuracies.append(mean_accuracy)

            accuracies_over_t[i].append([np.mean(accuracies), np.std(accuracies)])

    for i, m in enumerate(models):
        means, stds = [i[0] for i in accuracies_over_t[i]], [i[1] for i in accuracies_over_t[i]]

        fig = plt.figure(figsize=(25, 10), dpi=200)
        fig.suptitle('{}: Accuracies over Palindrome Lengths'.format(m), fontsize=40)
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(x=p_lengths, y=means, yerr=stds, fmt='-o', linewidth=4, color="g")
        ax.set_xticks(p_lengths)

        ax.set_xlabel('$Palindrome Length$', fontsize=28)
        ax.set_ylabel('$Accuracy$', fontsize=28)

        plt.savefig("part1/figures/{}_accuracies_over_t.png".format(m))
        plt.show()


def evaluate_accuracy(model, config):
    # device = torch.device(config.device)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    test_size = 8192

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, test_size, num_workers=1)

    batch_inputs, batch_targets = next(iter(data_loader))

    batch_inputs = torch.scatter(torch.zeros(*batch_inputs.size(), config.num_classes), 2,
                                 batch_inputs[..., None].to(torch.int64), 1).to(device)
    batch_targets = batch_targets.to(device)

    train_output = model.forward(batch_inputs)

    accuracy = torch.sum(torch.eq(torch.argmax(train_output, dim=1), batch_targets)).item() / train_output.size(0)

    return np.mean(accuracy)


def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    # TODO: Change all of these before handing in
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device(config.device)

    # Initialize params for models
    seq_length = config.input_length
    input_dim = config.input_dim
    num_hidden = config.num_hidden
    num_classes = config.num_classes

    print(seq_length, input_dim, num_classes, num_hidden)

    # Testing for convergence
    epsilon = 5e-4
    # minimal steps the model definitely trains, LSTM trains slower so needs more interations
    if seq_length < 30:
        if config.model_type == 'RNN':
            min_steps = 3000 if seq_length > 15 else 1000
        else:
            min_steps = 6000 if seq_length > 15 else 1500
    else:
        min_steps = config.train_steps

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(seq_length, input_dim, num_hidden, num_classes, device)
    else:
        model = LSTM(seq_length, input_dim, num_hidden, num_classes, device)

    model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Train losses
    losses = []
    # Train accuracies
    accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = torch.scatter(torch.zeros(*batch_inputs.size(), num_classes), 2,
                                     batch_inputs[..., None].to(torch.int64), 1).to(device)

        batch_targets = batch_targets.to(device)

        train_output = model.forward(batch_inputs)
        loss = criterion(train_output, batch_targets)

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        # Clip exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.eq(torch.argmax(train_output, dim=1), batch_targets)).item() / train_output.size(0)
        accuracies.append(accuracy)
        losses.append(loss.item())

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 100 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

            if step > min_steps and (np.absolute(np.mean(losses[-100:-2]) - losses[-1]) < epsilon):
                print("Convergence reached after {} steps".format(step))
                break

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    return model


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config)
    # local_experiments(config)
