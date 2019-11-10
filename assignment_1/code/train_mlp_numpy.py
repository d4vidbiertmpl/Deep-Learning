"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils as cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    acc = np.sum(np.equal(np.argmax(predictions, axis=1), np.argmax(targets, axis=1))) / predictions.shape[0]

    ########################
    # END OF YOUR CODE    #
    #######################

    return acc


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    if FLAGS.data_dir:
        DATA_DIR_DEFAULT = FLAGS.data_dir

    # Get negative slope parameter for LeakyReLU
    neg_slope = FLAGS.neg_slope

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    batch_size = FLAGS.batch_size

    criterion = CrossEntropyModule()

    cifar_data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    train_data = cifar_data['train']
    test_data = cifar_data['test']

    n_classes = train_data.labels.shape[1]
    n_inputs = np.prod(train_data.images.shape[1:])

    model = MLP(n_inputs, dnn_hidden_units, n_classes, neg_slope)

    x_test, y_test = test_data.images, test_data.labels
    x_test = np.reshape(x_test, (x_test.shape[0], n_inputs))

    # Train and Test losses
    losses = [[], []]
    # Train and Test accuracies
    accuracies = [[], []]

    # True iteration for plotting
    iterations = []

    for iteration in np.arange(FLAGS.max_steps):

        x, y = train_data.next_batch(batch_size)
        x = np.reshape(x, (batch_size, n_inputs))

        train_output = model.forward(x)
        g_loss = criterion.backward(train_output, y)
        model.backward(g_loss)

        for layer in model.net_layers:
            linear, _ = layer
            linear.params['weight'] -= FLAGS.learning_rate * linear.grads['weight']
            linear.params['bias'] -= FLAGS.learning_rate * linear.grads['bias']

        if iteration % FLAGS.eval_freq == 0 or iteration == FLAGS.max_steps - 1:
            iterations.append(iteration)

            # Second forward pass for test set
            test_output = model.forward(x_test)

            # Calculate losses
            train_loss = criterion.forward(train_output, y)
            losses[0].append(train_loss)

            test_loss = criterion.forward(test_output, y_test)
            losses[1].append(test_loss)

            # Calculate accuracies
            train_acc = accuracy(train_output, y)
            test_acc = accuracy(test_output, y_test)
            accuracies[0].append(train_acc)
            accuracies[1].append(test_acc)

            print("Iteration {}, Train loss: {}, Accuracy: {}".format(iteration, train_loss, train_acc))

    fig = plt.figure(figsize=(25, 10), dpi=200)
    fig.suptitle('Numpy MLP: Losses and Accuracies', fontsize=28)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(iterations, losses[0], linewidth=3, color="g", label="Train loss")
    ax1.plot(iterations, losses[1], linewidth=3, color="c", label="Test loss")
    ax2.plot(iterations, accuracies[0], linewidth=3, color="g", label="Train accuracy")
    ax2.plot(iterations, accuracies[1], linewidth=3, color="c", label="Test accuracy")

    ax1.set_xlabel('$Iteration$', fontsize=20)
    ax1.set_ylabel('$Loss$', fontsize=20)
    ax2.set_xlabel('$Iteration$', fontsize=20)
    ax2.set_ylabel('$Accuracy$', fontsize=20)

    ax1.legend(fontsize=20)
    ax2.legend(fontsize=20)

    plt.savefig("../figures/numpy_mlp.png")
    plt.show()


########################
# END OF YOUR CODE    #
#######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                        help='Negative slope parameter for LeakyReLU')
    FLAGS, unparsed = parser.parse_known_args()

    main()
