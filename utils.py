import numpy as np
from typing import *
from abc import ABC, abstractmethod


#激活函数
def sigmoid_vector(input_vector: np.typing.NDArray[np.float64]):
    return 1 / (1 + np.exp(-input_vector))

def relu_vector(input_vector: np.typing.NDArray[np.float64]):
    return np.maximum(0, input_vector)

def tanh_vector(input_vector: np.typing.NDArray[np.float64]):
    return (np.exp(input_vector) - np.exp(-input_vector)) / (np.exp(input_vector) + np.exp(-input_vector))

#激活函数的导数
def sigmoid_derivative_vector(input_vector: None, output_vector: np.typing.NDArray[np.float64] | None):
    return output_vector * (1 - output_vector)

def relu_derivative_vector(input_vector: np.typing.NDArray[np.float64] | None, output_vector: np.typing.NDArray[np.float64] | None):
    if input_vector is None:
        return (output_vector > 0).astype(np.float64)
    else:
        return (input_vector > 0).astype(np.float64)

def tanh_derivative_vector(input_vector: None, output_vector: np.typing.NDArray[np.float64] | None):
    return 1 - output_vector ** 2

class Optimizer(ABC):
    @abstractmethod
    def get_weights_delta(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        pass
    
    @abstractmethod
    def get_biases_delta(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        pass
    
class SGD(Optimizer):
    def __init__(self, learning_rate: int):
        self.learning_rate = learning_rate

    def get_weights_delta(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        return weights - gradient * self.learning_rate
    
    def get_biases_delta(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        return biases - gradient * self.learning_rate