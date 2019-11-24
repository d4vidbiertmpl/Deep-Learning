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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.device = device

        self.W_hx = nn.Parameter(torch.empty(input_dim, num_hidden), requires_grad=True)
        self.W_hh = nn.Parameter(torch.empty(num_hidden, num_hidden), requires_grad=True)
        self.W_ph = nn.Parameter(torch.empty(num_hidden, num_classes), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros(1, num_hidden), requires_grad=True)
        self.b_p = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)

        for w in [self.W_hx, self.W_hh, self.W_ph]:
            nn.init.xavier_normal_(w)

        self.h_init = torch.zeros(1, num_hidden).to(device)

        # For exercise 1.7
        self.h_states = []

    def forward(self, x):
        # Implementation here ...
        _h = self.h_init
        for t in range(self.seq_length):
            _h = (x[:, t, None] @ self.W_hx + _h @ self.W_hh + self.b_h).tanh()
        return _h @ self.W_ph + self.b_p

    # For exercise 1.7: decided to make an own function to not pollute the forward pass
    def analyze_hs_gradients(self, x):
        _h = torch.zeros(1, self.num_hidden, requires_grad=True).to(self.device)
        for t in range(self.seq_length):
            self.h_states.append(_h)
            _h = (x[:, t, None] @ self.W_hx + _h @ self.W_hh + self.b_h).tanh()
        return _h @ self.W_ph + self.b_p
