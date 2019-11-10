"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.std = 0.0001

        self.params = {'weight': self.std * np.random.randn(out_features, in_features),
                       'bias': np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias': np.zeros((out_features, 1))}

        self.weights_size = self.params['weight'].shape
        self.bias_size = self.params['bias'].shape

        self._x = np.zeros(0)
        self.out = np.zeros(0)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self._x = x

        self.out = (self.params['weight'] @ x.T + self.params['bias']).T

        ########################
        # END OF YOUR CODE    #
        #######################

        return self.out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout @ self.params['weight']

        self.grads['weight'] = dout.T @ self._x
        self.grads['bias'] = np.sum(dout, axis=0)[:, np.newaxis]

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class LeakyReLUModule(object):
    """
    Leaky ReLU activation module.
    """

    def __init__(self, neg_slope):
        """
        Initializes the parameters of the module.

        Args:
          neg_slope: negative slope parameter.

        TODO:
        Initialize the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.neg_slope = neg_slope
        self.out_mask = np.zeros(0)

        self.grad = None

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.out_mask = x <= 0
        # copy fixes a weird bug with the unit tests
        _x = np.copy(x)
        _x[self.out_mask] *= self.neg_slope

        ########################
        # END OF YOUR CODE    #
        #######################

        return _x

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = np.ones(self.out_mask.shape)
        dx[self.out_mask] = self.neg_slope
        dx *= dout

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        # Will be set by the forward pass
        self.out = np.zeros(0)

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        b = x.max(axis=1)[:, np.newaxis]
        y = np.exp(x - b)
        self.out = y / y.sum(axis=1)[:, np.newaxis]

        ########################
        # END OF YOUR CODE    #
        #######################

        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        _sm_diag = np.apply_along_axis(np.diag, 1, self.out)
        _sm_outer = self.out[:, :, np.newaxis] * self.out[:, np.newaxis, :]
        _g_softmax = _sm_diag - _sm_outer
        dx = np.einsum('ij, ijk -> ik', dout, _g_softmax)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.mean(np.sum(-y * np.log(x), axis=1))
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = - y / (x + 1e-12)
        dx = dx / y.shape[0]

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
