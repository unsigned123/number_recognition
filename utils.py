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

def none_vector(input_vector: np.typing.NDArray[np.float64]):
    return input_vector

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

def none_derivative_vector(input_vector: None, output_vector: np.typing.NDArray[np.float64] | None):
    return 1

####

class Optimizer(ABC):
    @abstractmethod
    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        pass
    
    @abstractmethod
    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        pass
    
class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        return weights - gradient * self.learning_rate
    
    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        return biases - gradient * self.learning_rate
    
class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.99):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.previous_velocity_weights = None
        self.previous_velocity_biases = None
    
    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_weights is None:
            self.previous_velocity_weights = np.zeros_like(gradient)
        velocity = self.momentum * self.previous_velocity_weights + self.learning_rate * gradient

        self.previous_velocity_weights = velocity
        return weights - velocity
    
    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_biases is None:
            self.previous_velocity_biases = np.zeros_like(gradient)
        velocity = self.momentum * self.previous_velocity_biases + self.learning_rate * gradient

        self.previous_velocity_biases = velocity
        return biases - velocity
    
class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, attenuation_rate: float = 0.99, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.attenuation_rate = attenuation_rate
        self.eps = eps

        self.previous_velocity_weights = None
        self.previous_velocity_biases = None
    
    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_weights is None:
            self.previous_velocity_weights = np.zeros_like(gradient)
        velocity = self.attenuation_rate * self.previous_velocity_weights + (1 - self.attenuation_rate) * (gradient ** 2)

        result = weights - self.learning_rate / ((self.eps + np.sqrt(velocity)) * gradient)

        self.previous_velocity_weights = velocity
        return result
    
    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_biases is None:
            self.previous_velocity_biases = np.zeros_like(gradient)
        velocity = self.attenuation_rate * self.previous_velocity_biases + (1 - self.learning_rate) * (gradient ** 2)

        result = biases - self.learning_rate / ((self.eps + np.sqrt(velocity)) * gradient)

        self.previous_velocity_biases = velocity
        return result
    
class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.eps = eps

        self.previous_velocity_weights = None
        self.previous_velocity_biases = None
    
    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_weights is None:
            self.previous_velocity_weights = np.zeros_like(gradient)
        velocity = self.previous_velocity_weights + (gradient ** 2)

        result = weights - self.learning_rate / ((self.eps + np.sqrt(velocity)) * gradient)

        self.previous_velocity_weights = velocity
        return result
    
    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        if self.previous_velocity_biases is None:
            self.previous_velocity_biases = np.zeros_like(gradient)
        velocity = self.previous_velocity_biases + (gradient ** 2)

        result = biases - self.learning_rate / ((self.eps + np.sqrt(velocity)) * gradient)

        self.previous_velocity_biases = velocity
        return result
    
class Adam(Optimizer):
    def __init__(self, learning_rate: float, momentum_attenuation: float = 0.99, learning_rate_attenuation: float = 0.99,
                 eps: float = 1e-8):
        self.learning_rate = learning_rate
        #beta_1
        self.momentum_attenuation = momentum_attenuation
        #beta_2
        self.learning_rate_attenuation = learning_rate_attenuation
        self.eps = eps

        self.previous_first_moment_estimate_weights = None
        self.previous_first_moment_estimate_biases = None
        self.previous_second_moment_estimate_weights = None
        self.previous_second_moment_estimate_biases = None

        self.step_counter_weights = 0
        self.step_counter_biases = 0

    def get_new_weights(self, weights: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        self.step_counter_weights += 1
        if self.previous_first_moment_estimate_weights is None or\
           self.previous_second_moment_estimate_weights is None:
            self.previous_first_moment_estimate_weights = np.zeros_like(gradient)
            self.previous_second_moment_estimate_weights = np.zeros_like(gradient)

        first_moment_estimate = self.momentum_attenuation * self.previous_first_moment_estimate_weights + \
                                (1 - self.momentum_attenuation) * gradient
        
        second_moment_estimate = self.learning_rate_attenuation * self.previous_second_moment_estimate_weights + \
                                (1 - self.learning_rate_attenuation) * (gradient ** 2)
        
        corrected_first_moment_estimate = first_moment_estimate / (1 - self.momentum_attenuation ** self.step_counter_weights)
        corrected_second_moment_estimate = second_moment_estimate / (1 - self.learning_rate_attenuation ** self.step_counter_weights)

        #保存上次的**有偏**估计
        self.previous_first_moment_estimate_weights = first_moment_estimate
        self.previous_second_moment_estimate_weights = second_moment_estimate

        return weights - self.learning_rate * corrected_first_moment_estimate / (np.sqrt(corrected_second_moment_estimate) + self.eps)

    def get_new_biases(self, biases: np.typing.NDArray[np.float64], gradient: np.typing.NDArray[np.float64]):
        self.step_counter_biases += 1
        if self.previous_first_moment_estimate_biases is None or\
           self.previous_second_moment_estimate_biases is None:
            self.previous_first_moment_estimate_biases = np.zeros_like(gradient)
            self.previous_second_moment_estimate_biases = np.zeros_like(gradient)

        first_moment_estimate = self.momentum_attenuation * self.previous_first_moment_estimate_biases + \
                                (1 - self.momentum_attenuation) * gradient
        
        second_moment_estimate = self.learning_rate_attenuation * self.previous_second_moment_estimate_biases + \
                                (1 - self.learning_rate_attenuation) * (gradient ** 2)
        
        corrected_first_moment_estimate = first_moment_estimate / (1 - self.momentum_attenuation ** self.step_counter_biases)
        corrected_second_moment_estimate = second_moment_estimate / (1 - self.learning_rate_attenuation ** self.step_counter_biases)

        #保存上次的**有偏**估计
        self.previous_first_moment_estimate_biases = first_moment_estimate
        self.previous_second_moment_estimate_biases = second_moment_estimate

        return biases - self.learning_rate * corrected_first_moment_estimate / (np.sqrt(corrected_second_moment_estimate) + self.eps)