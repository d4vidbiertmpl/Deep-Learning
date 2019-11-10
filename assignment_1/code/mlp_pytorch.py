"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          neg_slope: negative slope parameter for LeakyReLU

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.neg_slope = neg_slope

        self.linear_layers = nn.ModuleList()
        self.activations = []

        for i, h_no in enumerate(self.n_hidden):
            self.linear_layers.append(nn.Linear(n_inputs, h_no))
            self.activations.append(nn.LeakyReLU(negative_slope=neg_slope))
            n_inputs = h_no

        self.linear_layers.append(nn.Linear(n_inputs, n_classes))
        self.activations.append(None)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        for linear, activation in zip(self.linear_layers, self.activations):
            x = linear(x)
            if activation is not None:
                x = activation(x)

        ########################
        # END OF YOUR CODE    #
        #######################

        return x
