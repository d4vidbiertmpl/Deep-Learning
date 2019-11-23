# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel


def calc_accuracy(predictions, targets):
    corr_pred = torch.sum(torch.eq(torch.argmax(predictions, dim=1), targets)).item()
    return corr_pred / (targets.size(0) * targets.size(1))


################################################################################

def train(config):
    # Initialize the device which to run the model on
    # device = torch.device(config.device)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    seq_length = config.seq_length
    batch_size = config.batch_size
    lstm_num_hidden = config.lstm_num_hidden
    lstm_num_layers = config.lstm_num_layers
    dropout_keep_prob = config.dropout_keep_prob

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, seq_length)
    data_loader = DataLoader(dataset, batch_size, num_workers=1)

    vocab_size = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size, seq_length, vocab_size, lstm_num_hidden, lstm_num_layers, dropout_keep_prob,
                                device)
    model.to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, config.learning_rate_step, config.learning_rate_decay)

    # Train losses
    losses = []
    # Train accuracies
    accuracies = []

    print(batch_size, seq_length, vocab_size)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################

        # To onehot represetation
        batch_inputs = torch.scatter(torch.zeros(*batch_inputs.size(), vocab_size), 2, batch_inputs[..., None], 1).to(
            device)
        batch_targets = batch_targets.to(device)

        train_output = model.forward(batch_inputs)

        loss = criterion(train_output, batch_targets)
        accuracy = calc_accuracy(train_output, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(step)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                int(config.train_steps), config.batch_size, examples_per_second,
                accuracy, loss
            ))

        if step == config.sample_every:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
