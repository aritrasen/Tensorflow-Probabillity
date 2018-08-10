import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Bayesian neural network using tensorflow_probability")

    # Basic
    parser.add_argument(
        '--activation', '-a',
        type=str, default="relu",
        help="Activation function for all hidden layers.")

    parser.add_argument(
        '--layer_sizes', '-ls',
        type=lambda s: list(map(int, s.split(","))), default=[100, 50, 10],
        help="Comma-separated list denoting hidden units per layer.")

    parser.add_argument(
        '--learning_rate', '-lr',
        type=float, default=0.001,
        help="Initial learning rate.")

    parser.add_argument(
        '--training_epochs', '-ep',
        type=int, default=20,
        help="Number of epochs to run.")

    parser.add_argument(
        '--batch_size', '-bs',
        type=int, default=100,
        help="Batch size.")

    parser.add_argument(
        '--num_monte_carlo', '-ncarlo',
        type=int, default=50,
        help="Network draws to compute predictive probabilities.")

    parser.add_argument(
        '--seed',
        type=int, default=19931028,
        help="random seeds for tensorflow and numpy.")

    # Dropout or not
    parser.add_argument(
        "--keep_prob", "-kp",
        type=float, default=0.8,
        help="Probability of keeping neuron.")

    parser.add_argument(
        "--isdrop",
        type=bool, default=False,
        help="Whether to use dropout.")

    parser.add_argument(
        "--drop_pattern", "-dp",
        type=str, default="c",
        help="'e' for element-wise dropout (X*Z)W, \
        'c' for column-wise dropout.")

    # Alpha-BNN
    parser.add_argument(
        '--KLscale', '-s',
        type=float, default=1,
        help="Scale parameter for KL divergence regularization to priors.")

    # CNN
    parser.add_argument(
        '--inshape',
        type=lambda s: list(map(int, s.split(","))), default=[32, 32, 3],
        help="data input shape for convolution nnet. \
        Default CIFAR dataset: 32,32,3.")

    # Change prior & posterior
    parser.add_argument(
        '--priorstd',
        type=float, default=1,
        help="Std for prior Gaussian distributions.")

    parser.add_argument(
        '--poststd',
        type=float,
        help="Fix posterior")

    # Trial
    parser.add_argument(
        "--trial", "-t",
        type=str, default="1c",
        help="which experiment you're running now.")

    # Dataset
    parser.add_argument(
        "--data",
        type=str, default="cifar10",
        help="Dataset to choose: CIFAR10, MNIST\
        (case-insensitive). Default is CIFAR10.")

    # Dataset
    parser.add_argument(
        "--model", "-m",
        type=str, default="bnn",
        help="Model to run: bnn/snn. Deafault: bnn.")

    return parser
