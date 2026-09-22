import numpy as np
from layer import Layer

class Dense(Layer):
  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size

    self.weights, self.bias = self.HeInitialization()

  def HeInitialization(self):
    # define weights based on a normal distribution centered at 0 and variance of 2/n_in
    stddev = np.sqrt(2.0 / self.input_size)
    weights = np.random.randn(self.output_size, self.input_size) * stddev

    # initialize biases as 0
    biases = np.zeros((self.output_size, 1))
    return weights, biases
  
  def forward(self, input):
    self.input = input
    return self.weights @ self.input + self.bias
  
  def backward(self, output_gradient, batch_size, alpha):
    dW = (1 / batch_size) * output_gradient @ self.input.T
    db = (1 / batch_size) * np.sum(output_gradient, axis=1)
    db = db[:, None]

    self.weights -= alpha * dW
    self.bias -= alpha * db
    return self.weights.T @ output_gradient