import numpy as np
from activation import Activation

class ReLU(Activation):
  def __init__(self):
    relu = lambda x: np.maximum(0, x)
    relu_deriv = lambda x: x > 0
    super().__init__(relu, relu_deriv)