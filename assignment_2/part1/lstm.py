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

import torch
import torch.nn as nn


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.device = device

        # Makes more sense than ParameterList in this case
        self.lstm_cell = nn.ParameterDict()
        for layer in ["g{}", "i{}", "f{}", "o{}", "p{}"]:
            if layer == "p{}":
                self.lstm_cell[layer.format('h')] = nn.Parameter(torch.empty(num_hidden, num_classes),
                                                                 requires_grad=True)
                self.lstm_cell[layer.format('b')] = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)
            else:
                self.lstm_cell[layer.format('x')] = nn.Parameter(torch.empty(input_dim, num_hidden), requires_grad=True)
                self.lstm_cell[layer.format('h')] = nn.Parameter(torch.empty(num_hidden, num_hidden),
                                                                 requires_grad=True)
                if layer == "f{}":
                    self.lstm_cell[layer.format('b')] = nn.Parameter(torch.ones(1, num_hidden), requires_grad=True)
                else:
                    self.lstm_cell[layer.format('b')] = nn.Parameter(torch.zeros(1, num_hidden), requires_grad=True)

        for key in self.lstm_cell:
            if not key[-1] == 'b':
                # like suggested in the lecture
                nn.init.xavier_normal_(self.lstm_cell[key])

        self.h_init = torch.zeros(1, num_hidden).to(device)
        self.c_init = torch.zeros(1, num_hidden).to(device)

        # For exercise 1.7
        self.h_states = []

    def forward(self, x):
        # Implementation here ...
        _h, _c = self.h_init, self.c_init
        for t in range(self.seq_length):
            _g = (x[:, t] @ self.lstm_cell["gx"] + _h @ self.lstm_cell["gh"] + self.lstm_cell["gb"]).tanh()
            _i = (x[:, t] @ self.lstm_cell["ix"] + _h @ self.lstm_cell["ih"] + self.lstm_cell["ib"]).sigmoid()
            _f = (x[:, t] @ self.lstm_cell["fx"] + _h @ self.lstm_cell["fh"] + self.lstm_cell["fb"]).sigmoid()
            _o = (x[:, t] @ self.lstm_cell["ox"] + _h @ self.lstm_cell["oh"] + self.lstm_cell["ob"]).sigmoid()

            _c = _g * _i + _c * _f
            _h = _c.tanh() * _o

        return _h @ self.lstm_cell["ph"] + self.lstm_cell["pb"]

    # For exercise 1.7: decided to make an own function to not pollute the forward pass
    def analyze_hs_gradients(self, x):
        # Implementation here ...
        _h, _c = torch.zeros(1, self.num_hidden, requires_grad=True).to(self.device), self.c_init
        for t in range(self.seq_length):
            _g = (x[:, t] @ self.lstm_cell["gx"] + _h @ self.lstm_cell["gh"] + self.lstm_cell["gb"]).tanh()
            _i = (x[:, t] @ self.lstm_cell["ix"] + _h @ self.lstm_cell["ih"] + self.lstm_cell["ib"]).sigmoid()
            _f = (x[:, t] @ self.lstm_cell["fx"] + _h @ self.lstm_cell["fh"] + self.lstm_cell["fb"]).sigmoid()
            _o = (x[:, t] @ self.lstm_cell["ox"] + _h @ self.lstm_cell["oh"] + self.lstm_cell["ob"]).sigmoid()

            _c = _g * _i + _c * _f
            _h = _c.tanh() * _o

            _h.retain_grad()
            self.h_states.append((t, _h))

        return _h @ self.lstm_cell["ph"] + self.lstm_cell["pb"]
