"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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

        #
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.neg_slope = neg_slope

        self.net_layers = []

        for i, h_no in enumerate(self.n_hidden):
            self.net_layers.append((LinearModule(n_inputs, h_no),
                                    LeakyReLUModule(neg_slope)
                                    ))
            n_inputs = h_no

        self.net_layers.append((LinearModule(n_inputs, n_classes),
                                SoftMaxModule()
                                ))

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

        for layer in self.net_layers:
            linear, activation = layer
            x = linear.forward(x)
            x = activation.forward(x)

        ########################
        # END OF YOUR CODE    #
        #######################

        return x

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.net_layers[::-1]:
            linear, activation = layer
            dout = activation.backward(dout)
            dout = linear.backward(dout)

        ########################
        # END OF YOUR CODE    #
        #######################

        return
