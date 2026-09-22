import idx2numpy
import numpy as np
import time

# read data
def inputData(fileLocation):
  file_path = fileLocation
  data_raw = idx2numpy.convert_from_file(file_path)
  return data_raw

# flatten the data into a 2D array and return the transposed data set
def formatData(data_raw):
  reshape_data = data_raw.reshape(data_raw.shape[0], -1)
  data = reshape_data.T / 255
  return data

# one hot encode lables
def oneHotEncode(lables):
  one_hot = np.eye(10)[lables]
  return one_hot.T

A0 = formatData(inputData('mnist-dataset\\versions\\1\\train-images-idx3-ubyte\\train-images-idx3-ubyte'))
lables = inputData('mnist-dataset\\versions\\1\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte')
Y = oneHotEncode(lables)

labled_data = np.vstack((A0, Y, lables))

test_data = formatData(inputData('mnist-dataset\\versions\\1\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'))
test_lables = inputData('mnist-dataset\\versions\\1\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte')

# ***FORWARD PROPOGATION***
# Take an image and computes a prediction
# Input data is manipulated by weigths, biases and activation fucntions
# The prediction can then be compared to the target

# use He Initialization to set initial weights
# take the number of input and output nodes as parameters
def HeInitialization(n_in, n_out):
  # define weights based on a normal distribution centered at 0 and variance of 2/n_in
  stddev = np.sqrt(2.0 / n_in)
  weights = np.random.randn(n_out, n_in) * stddev

  # initialze biases as 0
  biases = np.zeros((n_out, 1))
  return weights, biases

def initializeWeights():
  W1, b1 = HeInitialization(784, 100)
  W2, b2 = HeInitialization(100, 10)
  return W1, b1, W2, b2

# ***ACTIVATION FUCNTIONS***
# Activation functions add complexity to each layer
# Without these functions each layer would be a linear combination of the last, effctively making it as if there were no hidden layers at all

# ReLU takes the maximum of 0 and input Z
def ReLU(Z):
  return np.maximum(0, Z)

# derivative of ReLU
def derivReLU(Z):
  return (Z > 0)

# softmax transforms logits into probabilites 
def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# *****

def forwardPropogation(W1, b1, W2, b2, A0):
  Z1 = W1 @ A0 + b1
  A1 = ReLU(Z1)
  Z2 = W2 @ A1 + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

# *****

def backwardPropogation(A0, Z1, A1, W2, A2, Y, batch_size):
  dZ2 = A2 - Y
  dW2 = (1 / batch_size) * dZ2 @ A1.T
  db2 = (1 / batch_size) * np.sum(dZ2, axis=1)
  db2 = db2[:, None]

  dZ1 = W2.T @ dZ2 * derivReLU(Z1)
  dW1 = (1 / batch_size) * dZ1 @ A0.T
  db1 = (1 / batch_size) * np.sum(dZ1, axis=1)
  db1 = db1[:, None]
  return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, lables):
    print(predictions[:10], lables[:10])
    return np.sum(predictions == lables) / lables.size * 100

def crossEntropyLoss(A2, Y, batch_size):
  C = np.log(np.sum((A2 * Y), axis=0))
  loss = - (1 / batch_size) * np.sum(C)
  return loss

def makePrediction(W1, b1, W2, b2, X):
  _, _ ,_, A2 = forwardPropogation(W1, b1, W2, b2, X)
  prediction = getPredictions(A2)
  return prediction

def testPredictions(X, Y, W1, b1, W2, b2):
  prediction = makePrediction(W1, b1, W2, b2, X)
  accuracy = getAccuracy(prediction, Y)
  return accuracy

def createMiniBatches(data, batch_size, indecies):
  shuffle_data = data[:, indecies]

  batch = np.empty((shuffle_data.shape[0], batch_size, 1))
  for i in range(shuffle_data.shape[1] // batch_size):
    batch = np.dstack((batch, shuffle_data[:, (batch_size * i):(batch_size * (i + 1))]))

  return batch[:, :, 1:]

def gradient_descent(epochs, alpha, batch_size, A0, labled_data):
  W1, b1, W2, b2 = initializeWeights()

  for i in range(epochs):
    start = time.perf_counter()
    print("Epoch: ", i)
    rng = np.random.default_rng()
    indicies = rng.permutation(A0.shape[1])
    labled_batch = createMiniBatches(labled_data, batch_size, indicies)
    A_batch = labled_batch[:A0.shape[0], :, :]
    Y_batch = labled_batch[A0.shape[0]:-1, :, :]
    lables_batch = labled_batch[-1, :, :]
    
    for j in range(A0.shape[1] // batch_size):
      Z1, A1, Z2, A2 = forwardPropogation(W1, b1, W2, b2, A_batch[:, :, j])
      dW1, db1, dW2, db2 = backwardPropogation(A_batch[:, :, j], Z1, A1, W2, A2, Y_batch[:, :, j], batch_size)
      W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    
      if (j % 100 == 0):
        print("Batch: ", j)
        print("Cost: ", crossEntropyLoss(A2, Y_batch[:, :, j], batch_size))
        print(f"Accuracy: {getAccuracy(getPredictions(A2), lables_batch[:, j])}%\n")
    
    accuracy = testPredictions(test_data, test_lables, W1, b1, W2, b2)
    print(f"Test accuracy: {accuracy}%\n")
    print(f"Time elapsed: {time.perf_counter() - start}s\n")

  return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(5, 0.1, 64, A0, labled_data)