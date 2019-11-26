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

from torch.utils.tensorboard import SummaryWriter

from part2.dataset import TextDataset
from part2.model import TextGenerationModel


def generate_from_model(model, dataset, T=30, sampling_type="greedy", tau=1.0, device=torch.device("cpu")):
    vocab_size = dataset.vocab_size
    sample_char = torch.randint(vocab_size, size=(1, 1), device=device)

    final_sequence = [sample_char.item()]

    char_sequence = sample_char
    for t in range(T):

        # Deciding one hot or embedding => decided for embedding
        # model_input = F.one_hot(char_sequence, vocab_size).type(torch.FloatTensor)
        model_input = char_sequence

        with torch.no_grad():
            train_output = model.forward(model_input)[:, :, 0][..., None]

        if sampling_type == "greedy":
            _pred = torch.argmax(train_output, dim=1)
        elif sampling_type == "use_temperature":
            # sm = torch.softmax(-tau * train_output, dim=1).view(-1)
            sm = torch.softmax(train_output / tau, dim=1).view(-1)
            _pred = torch.multinomial(sm, 1)[:, None]
        else:
            print("Unknown sampling type")
            break

        char_sequence = torch.cat((char_sequence, _pred), dim=1)
        final_sequence.append(_pred.item())

    return dataset.convert_to_string(final_sequence)


################################################################################

def train(config):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    writer = SummaryWriter()

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
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, config.learning_rate_step, config.learning_rate_decay)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################

        # To onehot represetation of input or embedding => decided for embedding
        batch_inputs = batch_inputs.to(device)
        # batch_inputs = F.one_hot(batch_inputs, vocab_size).type(torch.FloatTensor).to(device)
        batch_targets = batch_targets.to(device)

        train_output = model.forward(batch_inputs)

        loss = criterion(train_output, batch_targets)
        accuracy = torch.sum(torch.eq(torch.argmax(train_output, dim=1), batch_targets)).item() / (
                batch_targets.size(0) * batch_targets.size(1))

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Accuracy/train', accuracy, step)

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

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model

            base_string = "[{}] Train Step {:04d}/{:04d}, Sampling type: {}, Temperature: {}, Text: {} \n"

            # sample greedily
            model_sample = generate_from_model(model, dataset, device=device)
            greedy_string = base_string.format(datetime.now().strftime("%Y-%m-%d %H:%M"), step, int(config.train_steps),
                                               "Greedy", "none", model_sample)

            with open("greedy_samples.txt", "a") as text_file:
                text_file.write(greedy_string)

            for temperature in [0.5, 1.0, 2.0]:
                model_sample = generate_from_model(model, dataset, sampling_type="use_temperature", tau=temperature,
                                                   device=device)

                temp_string = base_string.format(datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                                                 int(config.train_steps), "use_temperature", temperature, model_sample)
                with open("temperature_samples.txt", "a") as text_file:
                    text_file.write(temp_string)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    writer.close()


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
