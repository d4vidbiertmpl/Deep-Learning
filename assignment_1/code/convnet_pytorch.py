"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torch import nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        super(ConvNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.output_size = 512

        self.net = nn.Sequential(
            # conv1
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # maxpool1
            nn.MaxPool2d(3, stride=2, padding=1),
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # maxpool2
            nn.MaxPool2d(3, stride=2, padding=1),
            # conv3a
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # conv3b
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # maxpool3
            nn.MaxPool2d(3, stride=2, padding=1),
            # conv4a
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv4b
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # maxpool4
            nn.MaxPool2d(3, stride=2, padding=1),
            # conv5a
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv5b
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # maxpool5
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Modeled this extra because I had problems with reshaping the last Conv Layer output
        self.last_linear = nn.Linear(self.output_size, n_classes)

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

        net_out = self.net(x)
        # view seems to be the best practice for reshaping in such a situation
        net_out_reshape = net_out.view(-1, self.output_size)
        out = self.last_linear(net_out_reshape)

        ########################
        # END OF YOUR CODE    #
        #######################

        return out
