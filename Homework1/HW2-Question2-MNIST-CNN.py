import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Input variables (place holders), images (input x) and y_ (labels)
x = tf.placeholder(tf.float32, shape=[None, 784])
images = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# First convolutional layer (conv1) and max_pooling layer (pool1)
# Number of filters is assumed to be 10
# w_1 is initialized by glorot_uniform (default function)
w_1 = tf.get_variable(name="w_1", shape=[3, 3, 1, 10])
b_1 = tf.get_variable(name='b_1', shape=[10], initializer=tf.constant_initializer(0.0))
conv_firstLayer = tf.nn.conv2d(images, w_1, strides=[1, 1, 1, 1], padding='SAME')

pre_activation = tf.nn.bias_add(conv_firstLayer, b_1)
conv1 = tf.nn.relu(pre_activation)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second convolutional layer (conv2) and max_pooling layer (pool2)
# Number of filters is assumed to be 10
# w_2 is initialized by glorot_uniform (default function)
w_2 = tf.get_variable(name="w_2", shape=[5, 5, 10, 10])
b_2 = tf.get_variable(name='b_2', shape=[10], initializer=tf.constant_initializer(0.0))
conv_secondLayer = tf.nn.conv2d(pool1, w_2, strides=[1, 1, 1, 1], padding='SAME')
pre_activation = tf.nn.bias_add(conv_secondLayer, b_2)
conv2 = tf.nn.relu(pre_activation)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Flatten the pool2 layer

flatten = tf.contrib.layers.flatten(pool2)

# Fully connected layer

w_fc1 = tf.get_variable('w_fc1', shape=[flatten.get_shape()[1], 1024])
b_fc1 = tf.get_variable('b_fc1', shape=[1024])

fc1 = tf.nn.relu(tf.matmul(flatten, w_fc1) + b_fc1)

# Output layer
w_fc2 = tf.get_variable('w_fc2', shape=[1024, 10])
b_fc2 = tf.get_variable('b_fc2', shape=[10])

y = tf.matmul(fc1, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Training model
# Use Adam as the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluate model
# Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# This is for part 3 and predicting the labels of 9 image examples
prediction_samples = tf.argmax(y, 1)

# Training Hyper-parameters
batch_size = 100
no_epochs = 1

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_examples = 0
  for i in range(550):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    num_examples += batch_size
    if num_examples % 10000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print('After training %d examples, training accuracy %g' % (num_examples, train_accuracy))

  print('Test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))

  # Predict labels for the first 9 examples of test images.
  predicted_labels = prediction_samples.eval(feed_dict={x: mnist.test.images[0:9]})
  # Plot the 9 images with their true and predicted labels
  dat = mnist.test.images[0:9]
  dat = np.reshape(dat, (9, 28, 28))
  labels = mnist.test.labels[0:9]
  labels = np.argmax(labels, axis=1)
  fig, ax = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='all')
  i = 0
  for row in ax:
      for col in row:
          plt.figure(1, figsize=(8, 6))
          col.imshow(dat[i])
          TrueLabel = labels[i]
          col.set_title('True Label {}, Predicted Label {}'.format(TrueLabel, predicted_labels[i]), fontsize=7)
          i += 1

  # Visualize all the feature maps (i.e., 10) of the first and second layer of convolutional layers for one image.
  # The image is the last example in the training process
  # Feature maps in the first layer
  image = np.reshape(batch[0][-1], [-1, 784])
  conv_firstLayer = conv_firstLayer.eval(feed_dict={x: image})
  conv_firstLayer = np.transpose(conv_firstLayer[0], [2, 0, 1])
  fig, ax = plt.subplots(nrows=5, ncols=2, sharex='all', sharey='all')
  i = 0
  for row in ax:
      for col in row:
          plt.figure(2)
          col.imshow(conv_firstLayer[i])
          col.set_title('Conv 1: Feature Map Number {}'.format(i), fontsize=7)
          i += 1
  fig.tight_layout()

  # Feature maps in the second layer
  conv_secondLayer = conv_secondLayer.eval(feed_dict={x: image})
  conv_secondLayer = np.transpose(conv_secondLayer[0], [2, 0, 1])
  fig, ax = plt.subplots(nrows=5, ncols=2, sharex='all', sharey='all')
  i = 0
  for row in ax:
      for col in row:
          plt.figure(3)
          col.imshow(conv_secondLayer[i])
          col.set_title('Conv 2: Feature Map Number {}'.format(i), fontsize=7)
          i += 1
  fig.tight_layout()

plt.show()




