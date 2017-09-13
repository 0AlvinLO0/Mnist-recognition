#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alvin
# Date: 2017-09-01
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data



# Using Tensorflow's default tools to fetch data, this is the same as what we did in the first homework assignment.
mnist = input_data.read_data_sets('./mnist', one_hot=True) 
# Random seed.
rseed = 42
batch_size = 200
lr = 1e-1
num_epochs = 50
num_hiddens = 500
num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]                
num_classes = mnist.train.labels.shape[1]
print('num_train=',num_train,' num_test=',num_test,' num_feats=',num_feats)



# Placeholders that should be filled with training pairs (x, y). Use None to unspecify the first dimension 
# for flexibility.
x = tf.placeholder(tf.float32, [None, num_feats], name="x")  
y = tf.placeholder(tf.int32, [None, num_classes], name="y")


# Model weights initialization.
# Your code here.
U1=np.sqrt(6.0/(num_feats+num_hiddens))
U2=np.sqrt(6.0/(num_hiddens+num_classes))

w1=tf.Variable(tf.random_uniform(shape=[num_feats,num_hiddens],minval=-U1,maxval=U1),name="w1")
b1=tf.Variable(tf.zeros(num_hiddens),name='b1')
w2=tf.Variable(tf.random_uniform(shape=[num_hiddens,num_classes],minval=-U2,maxval=U2),name="w2")
b2=tf.Variable(tf.zeros(num_classes),name='b2')



# logits is the log-probablity of each classes, forward computation.
# Your code here.
Y=tf.nn.relu(tf.matmul(x,w1)+b1)    ##matmul--matrix multiply
logits=tf.nn.relu(tf.matmul(Y,w2)+b2)



# Use TensorFlow's default implementation to compute the cross-entropy loss of classification.
# Your code here.
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss=tf.reduce_mean(cross_entropy)
# Build prediction function.
# Your code here.
pred=tf.nn.softmax(logits)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  ##return a list of [0,1,0...]
accuracy=tf.reduce_mean(tf.cast(correct_pred,"float"))       ##calculate the accuracy
# Use TensorFlow's default implementation for optimziation algorithm. 
# Your code here.
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(loss)




# Start training!
num_batches = num_train / batch_size
losses = []
train_accs, valid_accs = [], []
time_start = time.time()
with tf.Session() as sess:
    # Before evaluating the graph, we should initialize all the variables.
    sess.run(tf.global_variables_initializer())
    for i in range(0,num_epochs):
        # Each training epoch contains num_batches of parameter updates.
        total_loss = 0.0
        for j in range(0,int(num_batches)):
            # Fetch next mini-batch of data using TensorFlow's default method.
            x_batch, y_batch = mnist.train.next_batch(batch_size)
             ###sess.run(optimizer,feed_dict={x:x_batch,y:y_batch})
            # Note that we also need to include optimizer into the list in order to update parameters, but we 
            # don't need the return value of optimizer.
            _, loss_batch = sess.run([optimizer,loss],feed_dict={x:x_batch,y:y_batch})  ##run() feed_dict is used to input data
            total_loss += loss_batch
            
        # Compute training set and validation set accuracy after each epoch.
        train_acc = sess.run([accuracy],feed_dict={x:mnist.train.images, y:mnist.train.labels})
        valid_acc = sess.run([accuracy],feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
        losses.append(total_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        print "Number of iteration: {}, total_loss = {}, train accuracy = {}, validation accuracy = {}".format(i, total_loss, train_acc, valid_acc)
    # Evaluate the test set accuracy at the end.
    test_acc = sess.run([accuracy],feed_dict={x:mnist.test.images, y:mnist.test.labels})
time_end = time.time()
print "Time used for training = {} seconds.".format(time_end - time_start)
print "MNIST image classification accuracy on test set = {}".format(test_acc)




# Plot the losses during training.
plt.figure()
plt.title("MLP-784-500-10 with TensorFlow")
plt.plot(losses, "b-o", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Cross-entropy")
plt.show()
