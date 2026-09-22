import idx2numpy
import numpy as np
import time

from dense import Dense
from activations import ReLU

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

def createMiniBatches(data, batch_size, indecies):
  shuffle_data = data[:, indecies]

  batch = np.empty((shuffle_data.shape[0], batch_size, 1))
  for i in range(shuffle_data.shape[1] // batch_size):
    batch = np.dstack((batch, shuffle_data[:, (batch_size * i):(batch_size * (i + 1))]))

  return batch[:, :, 1:]

# softmax transforms logits into probabilites 
def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def crossEntropyLoss(A2, Y, batch_size):
  C = np.log(np.sum((A2 * Y), axis=0))
  loss = - (1 / batch_size) * np.sum(C)
  return loss

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, lables):
    print(predictions[:10], lables[:10])
    return np.sum(predictions == lables) / lables.size * 100

def makePrediction(X):
  output = X
  for layer in network:
    output = layer.forward(output)
  output = softmax(output)

  prediction = getPredictions(output)
  return prediction

def testPredictions(X, Y):
  prediction = makePrediction(X)
  accuracy = getAccuracy(prediction, Y)
  return accuracy

network = [
  Dense(784, 100),
  ReLU(),
  Dense(100, 10)
]

def gradient_descent(epochs, alpha, batch_size, A0, labled_data):
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
      output = A_batch[:, :, j]
      for layer in network:
        output = layer.forward(output)
      output = softmax(output)

      grad = output - Y_batch[:, :, j]
      for layer in reversed(network):
        grad = layer.backward(grad, batch_size, alpha)
    
      if (j % 100 == 0):
        print("Batch: ", j)
        print("Cost: ", crossEntropyLoss(output, Y_batch[:, :, j], batch_size))
        print(f"Accuracy: {getAccuracy(getPredictions(output), lables_batch[:, j])}%\n")
    
    accuracy = testPredictions(test_data, test_lables)
    print(f"Test accuracy: {accuracy}%\n")
    print(f"Time elapsed: {time.perf_counter() - start}s\n")

gradient_descent(5, 0.1, 64, A0, labled_data)