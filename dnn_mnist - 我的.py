#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alvin
# Date: 2017-08-28
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# Load the MNIST dataset from the official website.
mnist = input_data.read_data_sets("mnist/", one_hot=True)

num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]


# Set hyperparameters of MLP.
rseed = 42
batch_size = 200
lr = 1e-1
num_hiddens = 500
num_epochs = 50


# Initialize model parameters, sample W ~ [-U, U], where U = sqrt(6.0 / (fan_in + fan_out)).
np.random.seed(rseed)
# Your code here to create model parameters globally.

u1=np.sqrt(6.0/(num_feats+num_hiddens))    
w1=(np.random.rand(num_feats,num_hiddens)-0.5)*2.0* u1
b1=np.zeros(num_hiddens)

u2=np.sqrt(6.0/(num_hiddens+num_classes))
w2=(np.random.rand(num_hiddens,num_classes)-0.5)*2.0* u2
b2=np.zeros(num_classes)


# Used to store the gradients of model parameters.
dw1 = np.zeros((num_feats, num_hiddens))
db1 = np.zeros(num_hiddens)
dw2 = np.zeros((num_hiddens, num_classes))
db2 = np.zeros(num_classes)


# Helper functions.
def ReLU(inputs):
    """

    Compute the ReLU: max(x, 0) nonlinearity.
    """
    inputs[inputs< 0.0]=0.0
    return inputs
  

def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.
    """
    probs=np.exp(inputs)
    probs /=np.sum(probs,axis=1)[:, np.newaxis]  #zhi you zhe yige wenti le ?? the meaning of np.newaxis
    return probs
   


def forward(inputs):
    """
    Forward evaluation of the model.
    """
    h1=ReLU(np.dot(inputs, w1)+ b1)
    h2=np.dot(h1, w2)+ b2
    
    return (h1, h2), softmax(h2)


def backward(probs, labels, (x, h1, h2)):
    """
    Backward propagation of errors.
    """
    n = probs.shape[0]
    a2 = probs - labels
    a1 = np.dot(a2, w2.T)
    a1[h1 < 0.0] = 0.0
    dw2[:] = np.dot(h1.T, a2) / n ##Have n images in total,to calculate the mean value
    db2[:] = np.mean(a2, axis = 0)
    
    dw1[:] = np.dot(x.T, a1) / n   
    db1[:] = np.mean(a1, axis = 0)   
    


def predict(probs):
    """
    Make predictions based on the model probability.
    """
    return np.argmax(probs,axis=1)



def evaluate(inputs, labels):
    """
    Evaluate the accuracy of current model on (inputs, labels).
    """
    _, probs=forward(inputs)
    pred= predict(probs)
    trues=np.argmax(labels,axis=1)
    return np.mean(pred==trues)
 

# Training using stochastic gradient descent.
time_start = time.time()
num_batches = num_train / batch_size
train_accs, valid_accs = [], []
for i in xrange(num_epochs): ##循环步数
    for j in xrange(num_batches):  ##循环批量数
        # Fetch the j-th mini-batch of the data.
        insts = mnist.train.images[batch_size * j: batch_size * (j+1), :]
        labels = mnist.train.labels[batch_size * j: batch_size * (j+1), :]
        # Forward propagation.
        (h1, h2), probs=forward(insts)
        
        # Backward propagation.
        backward(probs,labels,(insts, h1, h2))
        
        # Gradient update.
        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2
        
    # Evaluate on both training and validation set.
    train_acc = evaluate(mnist.train.images, mnist.train.labels)
    valid_acc = evaluate(mnist.validation.images, mnist.validation.labels)
    train_accs.append(train_acc) #append向列表尾部添加train_acc元素
    valid_accs.append(valid_acc)##
    print "Number of iteration: {}, classification accuracy on training set = {}, classification accuracy on validation set: {}".format(i, train_acc, valid_acc)
time_end = time.time()
# Compute test set accuracy.
acc = evaluate(mnist.test.images, mnist.test.labels)
print "Final classification accuracy on test set = {}".format(acc)
print "Time used to train the model: {} seconds.".format(time_end - time_start)

# Plot classification accuracy on both training and validation set for better visualization.
plt.figure()
plt.plot(train_accs, "bo-", linewidth=2)
plt.plot(valid_accs, "go-", linewidth=2)
plt.legend(["training accuracy", "validation accuracy"], loc=4)
plt.grid(True)
plt.show()
