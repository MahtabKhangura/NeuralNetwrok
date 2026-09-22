import numpy as np
from layer import Layer

class Activation(Layer):
  def __init__(self, activation, activation_deriv):
    self.activation = activation
    self.activation_deriv = activation_deriv

  def forward(self, input):
    self.input = input
    return self.activation(self.input)
  
  def backward(self, output_gradient, batch_size, alpha):
    return output_gradient * self.activation_deriv(self.input)