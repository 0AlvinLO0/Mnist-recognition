'''Trains a simple convnet on the MNIST dataset.
it is easy to get over 99.6% for testing accuracy
after a few epochs. And thre is still a lot of margin for parameter tuning,
e.g. different initializations, decreasing learning rates when the accuracy
stops increasing, or using model ensembling techniques, etc.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#print the former 6 train images below
import matplotlib.pyplot as plt
plt.subplot(321)
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))

plt.subplot(322)
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))

plt.subplot(323)
plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))

plt.subplot(324)
plt.imshow(x_train[3],cmap=plt.get_cmap('gray'))

plt.subplot(325)
plt.imshow(x_train[4],cmap=plt.get_cmap('gray'))

plt.subplot(326)
plt.imshow(x_train[5],cmap=plt.get_cmap('gray'))
plt.show()


if K.image_data_format() == 'channels_first':
     ##reshape it so that it is suitable for use training a CNN.
    ##In Keras, the layers used for two-dimensional convolutions expect pixel values with the dimensions [pixels][width][height].
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols).astype('float32')
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
    input_shape = (img_rows, img_cols, 1)
    

# In order to speed up the convergence, we may normalize the input values
# so that they are in the range of (0, 1) for (-1, 1)
# Your code here.
X_train =x_train/255.0
X_test =x_test/255.0


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices, e.g. "1" ==> [0,1,0,0,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.yo_categorical(y_test,num_classes)

model = Sequential()


# Please build up a simple ConvNets by stacking a few conovlutioanl layers (kenel size with 3x3
# is a good choice, don't foget using non-linear activations for convolutional layers),
# max-pooling layers, dropout layers and dense/fully-connected layers.
model.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))

model.add(Dropout(0.2))##drop 20% data
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
          

# complete the loss and optimizer
# Hints: use the cross-entropy loss, optimizer could be SGD or Adam, RMSProp, etc.
# Feel free to try different hyper-parameters.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),  ##use ADAM here
              metrics=['accuracy'])


# Extra Points 1: use data augmentation in the progress of model training.
# Note that data augmentation is a practical technique for reducing overfitting.
# Hints: you may refer to https://keras.io/preprocessing/image/
# here we use random ratation


###An augmented image generator in keras can be created as follows:
#datagen = ImageDataGenerator()

###After you have created and configured your ImageDataGenerator, you must fit it on your data.
###You can do this by calling the fit() function on the data generator and pass it your training dataset.
#datagen.fit(train)

###We can configure the batch size and prepare the data generator and get batches of images by
###calling the flow() function.
#X_batch, y_batch = datagen.flow(train, train, batch_size=32)

###Finally we can make use of the data generator,we must call the fit_generator() function and
###pass in the data generator and the desired length of an epoch as well as the total number of epochs of training
#fit_generator(datagen, samples_per_epoch=len(train), epochs=100)

datagen=ImageDataGenerator(rotation_range=60)
datagen.fit(x_train)
x_batch,y_batch=datagen.flow(x_train,y_train,batch_size=128)

plt.subplot(321)
plt.imshow(x_batch[0],cmap=plt.get_cmap('gray'))

plt.subplot(322)
plt.imshow(x_batch[1],cmap=plt.get_cmap('gray'))

plt.subplot(323)
plt.imshow(x_batch[2],cmap=plt.get_cmap('gray'))

plt.subplot(324)
plt.imshow(x_batch[3],cmap=plt.get_cmap('gray'))

plt.subplot(325)
plt.imshow(x_batch[4],cmap=plt.get_cmap('gray'))

plt.subplot(326)
plt.imshow(x_batch[5],cmap=plt.get_cmap('gray'))
plt.show()

fit_generator(datagen,samples_per_epoch=len(x_train),epochs=20)
# Extra Points 2: use K-Fold cross-validation for ensembling k models,
# i.e. (1) split the whole training data into K folds;
#      (2) train K models based on different training data;
#      (3) when evaludating the testing data, averaging over K model predictions as final output.
# The codes may look like:
#   for i in range(K):
#       x_train, y_train = ...
#       model_i = train(x_train , y_train)


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


