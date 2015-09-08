
### IMCOMPLETE CHAPTER ###

from __future__ import division
import math
from linear_algebra import dot


def step_function(x):
    return 1 if x >= 0 else 0


def perceptrons_outputs(weights, bias, x):
    """returns 1 if perceptron 'fires', 0 if not"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def sigmoid(t):
    return 1 / (1 + math.exp(-t))


def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

