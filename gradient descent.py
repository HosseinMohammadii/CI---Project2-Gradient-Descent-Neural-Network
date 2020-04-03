import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Define a Neuron as a class
class Neuron:

    def __init__(self, afType):
        self.weights = np.random.standard_normal(size=2)
        self.bias = np.random.standard_normal(size=None)
        self.afType = afType

    def activate(self, input):
        if self.afType == 'sigmoid':
            output = np.sum(np.dot(self.weights, input)) + self.bias

            return sigmoid(output)


import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Define our NeuralNetwork as a class


class HwNetwork:

    def __init__(self, train_data, test_data, type, learningRate):
        self.teData = test_data
        self.trData = train_data
        self.lr = learningRate
        self.type = type
        self.neurons = []
        self.out = []

        if self.type == 1:
            print("type 1  0")
            self.neurons.append(Neuron('sigmoid'))
        elif self.type == 2:
            print("type 2  0")
            # First layer Neuron 1
            self.neurons.append(Neuron('sigmoid'))

            # First layer Neuron 2
            self.neurons.append(Neuron('sigmoid'))

            # Second layer Neuron 1
            self.neurons.append(Neuron('sigmoid'))

    def learn(self, epoch):
        if self.type == 1:
            print("type 1  1")
            for i in range(0, epoch):
                # for j in range(0, 2):
                    grad = [0, 0]
                    gradB = 0
                    for dat in self.trData:
                        y = self.neurons[0].activate(dat[0:2])
                        cost = 1 / 2 * (y - dat[2]) ** 2
                        grad += self.gradDescent(y, dat, 0)
                        # print(grad)
                        # print("------------------------------------------")

                        gradB += self.gradDescent(y, dat, 1)

                # Updating weights as well as bias
                    self.neurons[0].bias -= self.lr * gradB * (1/140)
                    for k in range(0, 2):
                        # print(grad[k])
                        self.neurons[0].weights[k] = self.neurons[0].weights[k] - self.lr * grad[k] * (1/140)
        if self.type == 2:
            print("type 2  1")
            for i in range(0, epoch):
                gradW = [0, 0]
                gradV = [0, 0]
                gradU = [0, 0]
                gradB0 = 0
                gradB1 = 0
                gradB2 = 0
                for j in range(0, 3):

                    for dat in self.trData:
                        z0 = self.neurons[0].activate(dat[0:2])
                        z1 = self.neurons[1].activate(dat[0:2])
                        z = [z0, z1]
                        y  = self.neurons[2].activate(z)
                        y0 = dat[2]

                        cost = 1 / 2 * (y - y0) ** 2
                        if j == 0:
                            gradW += self.gradDescentW(y, y0, z0, dat, 0)
                            gradB0 += self.gradDescentW(y, y0, z0, dat, 1)
                        elif j == 1:
                            gradV += self.gradDescentV(y, y0, z1, dat, 0)
                            gradB1 += self.gradDescentV(y, y0, z1, dat, 1)
                        else:
                            gradU += self.gradDescentU(y, y0,  z, 0)
                            gradB2 += self.gradDescentU(y, y0, z, 1)

                # Updating weights as well as bias
                self.neurons[0].bias -= self.lr * gradB0 * (1/140)
                self.neurons[1].bias -= self.lr * gradB1 * (1/140)
                self.neurons[2].bias -= self.lr * gradB2 * (1/140)
                for k in range(0, 2):
                    self.neurons[0].weights[k] -= self.lr * gradW[k] * (1/140)
                    self.neurons[1].weights[k] -= self.lr * gradV[k] * (1/140)
                    self.neurons[2].weights[k] -= self.lr * gradU[k] * (1/140)

    def gradDescent(self, y, dat, f):
        X = dat[0:2]

        if f == 0:
            cGrad = (y - dat[2]) * (y * (1 - y)) * X
            # print(X)
            # print(cGrad)
        else:
            cGrad = (y - dat[2]) * (y * (1 - y))


        return cGrad

    def gradDescentW(self, y, y0, z0, dat, f):
        X = dat[0:2]
        if f == 0:
            cGrad = (y - y0) * (y * (1 - y)) * self.neurons[2].weights[0] * z0 * (1 - z0) * X
        else:
            cGrad = (y - y0) * (y * (1 - y)) * self.neurons[2].weights[0] * z0 * (1 - z0)

        return cGrad

    def gradDescentV(self, y, y0, z1, dat, f):
        X = dat[0:2]
        if f == 0:
            cGrad = (y - y0) * (y * (1 - y)) * self.neurons[2].weights[1] * z1 * (1 - z1) * X
        else:
            cGrad = (y - y0) * (y * (1 - y)) * self.neurons[2].weights[1] * z1 * (1 - z1)

        return cGrad

    def gradDescentU(self, y, y0, dat, f):
        Z = dat[0:2]
        cGrad = []
        if f == 0:
            cGrad.append((y - y0) * (y * (1 - y)) * Z[0])
            cGrad.append((y - y0) * (y * (1 - y)) * Z[1])
        else:
            cGrad = (y - y0) * (y * (1 - y))

        return cGrad

    def output(self):
        if self.type == 1:
            print("type 1  2")
            for tdat in self.teData:
                if self.neurons[0].activate(tdat[0:2]) >= 0.5:
                    self.out.append(1)
                else:
                    self.out.append(0)
        if self.type == 2:
            print("type 2  2")
            for tdat in self.teData:
                z0 = self.neurons[0].activate(tdat[0:2])
                z1 = self.neurons[1].activate(tdat[0:2])
                z = [z0, z1]
                y = self.neurons[2].activate(z)
                if y >= 0.5:
                    self.out.append(1)
                else:
                    self.out.append(0)

        return self.out

    def validate(self):
        t = 0
        i = 0
        for label in self.out:
            if label == self.teData[i][2]:
                t += 1
            i += 1

        return (t / len(self.teData) * 100)


# Main Program

import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# Loading data
fName = 'data.csv'
df = pd.read_csv(fName)

# Importing the data

X0 = []
X1 = []
Y0 = []

# Forming x&y axis appending the first data in each column
X0.append(0.20902463509260166)
for item in df['0.20902463509260166']:
    X0.append(item)

X1.append(0.2720141007611026)
for item in df['0.2720141007611026']:
    X1.append(item)

Y0.append(0)
for item in df['0']:
    Y0.append(item)

data = np.vstack((X0, X1, Y0)).T


# Scattering the data

def scatter(data):
    plt.figure()

    f1 = f2 = 0
    for dat in data:
        if dat[2] == 1:
            if f1 == 0:
                plt.scatter(dat[0], dat[1], color='red', label='Label=1')
                f1 = 1
            else:
                plt.scatter(dat[0], dat[1], color='red')
        else:
            if f2 == 0:
                plt.scatter(dat[0], dat[1], color='blue', label='Label=0')
                f2 = 1
            else:
                plt.scatter(dat[0], dat[1], color='blue')

    plt.title('Scattered data!')
    plt.legend(loc="upper left")
    plt.show()




scatter(data)

# Shuffling data
shuffledData = shuffle(data)

# Dividing into 2 subsets(30 percent of data for test)
train_data = shuffledData[0:int(0.7 * len(shuffledData))]
test_data = shuffledData[int(0.7 * len(shuffledData)):len(shuffledData)]

# Train the Neural Network

# Type=1 --> Single Neuron
myNetwork = HwNetwork(train_data, test_data, 1, 3)
myNetwork.learn(3000)
labels = myNetwork.output()
print('first Validation Percent = {}'.format(myNetwork.validate()))

plt.figure()
f1 = f2 = 0
for i in range(0, len(labels)):
    if labels[i] == 1:
        if f1 == 0:
            plt.scatter(test_data[i][0], test_data[i][1], color='red', label='Label=1')
            f1 = 1
        else:
            plt.scatter(test_data[i][0], test_data[i][1], color='red')
    else:
        if f2 == 0:
            plt.scatter(test_data[i][0], test_data[i][1], color='blue', label='Label=0')
            f2 = 1
        else:
            plt.scatter(test_data[i][0], test_data[i][1], color='blue')

plt.title('Scattered data!')
plt.legend(loc="upper left")
plt.show()

# Type=2 --> Single Neuron
myNetwork2 = HwNetwork(train_data, test_data, 2, 3)
myNetwork2.learn(3000)
labels2 = myNetwork2.output()
print('second Validation Percent = {}'.format(myNetwork2.validate()))

plt.figure()
f1 = f2 = 0
for i in range(0, len(labels)):
    if labels2[i] == 1:
        if f1 == 0:
            plt.scatter(test_data[i][0], test_data[i][1], color='red', label='Label=1')
            f1 = 1
        else:
            plt.scatter(test_data[i][0], test_data[i][1], color='red')
    else:
        if f2 == 0:
            plt.scatter(test_data[i][0], test_data[i][1], color='blue', label='Label=0')
            f2 = 1
        else:
            plt.scatter(test_data[i][0], test_data[i][1], color='blue')

plt.title('Scattered data!')
plt.legend(loc="upper left")
plt.show()