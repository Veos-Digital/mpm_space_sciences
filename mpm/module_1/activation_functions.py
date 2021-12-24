import numpy as np
import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_prime(x):
    ex = torch.exp(-x)
    return ex / (1 + ex)**2


def identity(x):
    return x


def identity_prime(x):
    return 1


def tanh(x):
    raise ValueError('Please, implement me!')


def tanh_prime(y):
    raise ValueError('Please, implement me!')


def relu(x):
    raise ValueError('Please, implement me!')


def relu_prime(y):
    raise ValueError('Please, implement me!')


ACTIVATION_DICT = {
    "sigmoid": {"func": sigmoid, "func_prime": sigmoid_prime},
    "relu": {"func": relu, "func_prime": relu_prime},
    "identity": {"func": identity, "func_prime": identity_prime}
}
