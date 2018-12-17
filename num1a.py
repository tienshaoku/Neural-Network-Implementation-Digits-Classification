import os, sys, struct, warnings, random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict    # used for data structure
from mnist import load_mnist           # load mnist.py under the same directary
sys.path.append(os.pardir)


class Sigmoid:
    def __init(self):
        self.out = None

    def forward(self, x):
        out      = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx       = dout * (1.0 - self.out) * self.out
        return dx


class MatrixMulti:
    def __init__(self, W, b):
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x  = x
        out     = np.dot(x, self.W)+ self.b
        return out

    def backward(self, dout):
        dx      = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx


class Relu:
    def __init(self):
        self.mask      = None

    def forward(self, x):
        self.mask      = (x <= 0)
        out            = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask]= 0
        dx             = dout
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss  = None
        self.y     = None
        self.t     = None

    def forward(self, x, t):
        self.t     = t
        self.y     = softmax(x)
        self.loss  = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx         = (self.y - self.t) / batch_size
        return dx


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1']           = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1']           = np.zeros(hidden_size)
        self.params['W2']           = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2']           = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['MatrixMulti1'] = MatrixMulti(self.params['W1'], self.params['b1'])
        self.layers['Relu1']        = Relu()
        self.layers['MatrixMulti2'] = MatrixMulti(self.params['W2'], self.params['b2'])

        self.lastLayer              = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y     = self.predict(x)
        y     = np.argmax(y, axis=1)
        if t.ndim != 1 : 
            t = np.argmax(t, axis = 1)
            
        accuracy  = np.sum(y == t) / float(x.shape[0])
        return accuracy 
    

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout     = 1
        dout     = self.lastLayer.backward(dout)

        layers   = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads    = {}
        grads['W1'], grads['b1'] = self.layers['MatrixMulti1'].dW, self.layers['MatrixMulti1'].db
        grads['W2'], grads['b2'] = self.layers['MatrixMulti2'].dW, self.layers['MatrixMulti2'].db

        return grads




def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x      = x.T
        x      = x - np.max(x, axis=0)
        y      = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t      = t.reshape(1, t.size)
        y      = y.reshape(1, y.size)

    if t.size == y.size:
        t      = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size




(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network         = NeuralNetwork(input_size=784, hidden_size=256, output_size=10)

iters_num       = 10000
train_size      = x_train.shape[0]
test_size       = x_test.shape[0]
batch_size      = 64
learning_rate   = 0.1

train_loss_list = []
test_loss_list  = []
train_acc_list  = []
test_acc_list   = []

iter_per_epoch  = max(train_size / batch_size, 1)


for i in range(iters_num):
    train_batch       = np.random.choice(train_size, batch_size)
    test_batch        = np.random.choice(test_size, batch_size)
    x_train_batch     = x_train[train_batch]
    t_train_batch     = t_train[train_batch]
    x_test_batch      = x_test[test_batch]
    t_test_batch      = t_test[test_batch]
 
    grad              = network.gradient(x_train_batch, t_train_batch)
    
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss_train = network.loss(x_train_batch, t_train_batch)
    loss_test  = network.loss(x_test_batch, t_test_batch)
    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)

    if i % iter_per_epoch == 0:
        train_acc  = network.accuracy(x_train, t_train)
        test_acc   = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train accuracy:", train_acc, ';', "test accuracy:", test_acc)


loss_train = np.array(train_loss_list)
print('loss_train:', loss_train)
plt.plot(np.arange(10000), loss_train, linewidth = 0.5, markersize = 0.5)
plt.title('loss curve: train')
plt.savefig('num1a_loss_curve_train.png')
plt.show()

loss_test  = np.array(test_loss_list)
print("loss_test", loss_test)
plt.plot(np.arange(10000), loss_test, linewidth = 0.5, markersize = 0.5)
plt.title('loss curve: test')
plt.savefig('num1a_loss_curve_test.png')
plt.show()



