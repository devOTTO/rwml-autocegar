"""Minimal vendored helpers from TSB-AD (TSB_AD/utils/utility.py) needed by the
reproduction models. Kept tiny and self-contained so deepant/ · rw/ · autocegar/
run without the full TSB-AD package on the path."""
from torch import nn


def get_activation_by_name(name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
    }
    if name in activations:
        return activations[name]
    raise ValueError(name, "is not a valid activation function")
