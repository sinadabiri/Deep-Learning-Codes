import csv
import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf


input = open('../HW3/text.txt').read().lower().split()
vocab = sorted(list(set(input)))
# Map words in the vocabulary to integers
word_to_int = dict((word, int) for int, word in enumerate(vocab))

# create training data set
seq_length = 5  # choose from this list [2, 5, 10]
X_data = []
Y_data = []
for i in range(0, len(input) - seq_length, 1):
    seq_in = input[i:i + seq_length]
    seq_out = input[i + seq_length]
    X_data.append([word_to_int[word] for word in seq_in])
    Y_data.append(word_to_int[seq_out])

# Scale (normalize) the X_data to integers between 0 and 1
X = np.reshape(X_data, (len(X_data), seq_length, 1))/float(len(vocab) - 1)
# Convert Y_data to one-hot encoding
encoder = OneHotEncoder(n_values=len(vocab))
Y = encoder.fit_transform(np.reshape(Y_data, (len(Y_data), 1))).toarray()

# Create the model in TensorFlow
# Input variables (place holders), input x (sequence of 2 or 5 or 10 words) and y_ (labels)
x = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, 1])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, len(vocab)])

Units = [512]  # For 1-layer LSTM, UNits is [512]. For 2-layer LSTM, UNits is [512, 512]. For 3-layer LSTM, UNits is [512, 412, 512]
lstm_layers = [tf.contrib.rnn.BasicLSTMCell(size) for size in Units]
LSTM = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
RNNCell = tf.contrib.rnn.DropoutWrapper(cell=LSTM, output_keep_prob=0.50)
outputs, _ = tf.nn.dynamic_rnn(RNNCell, x, dtype=tf.float32)

# output layer for classification task
weight = tf.Variable(tf.truncated_normal([Units[-1], len(vocab)]))
# weight = tf.get_variable('w_output', shape=[lstmUnits, numClasses])
bias = tf.Variable(tf.constant(0.1, shape=[len(vocab)]))
# bias = tf.get_variable('bias', shape=[numClasses], initializer=tf.constant_initializer(0.1))
outputs = tf.transpose(outputs, [1, 0, 2])
last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
prediction = (tf.matmul(last_output, weight) + bias)

# Training model
# Use Adam as the optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Evaluate model
# Accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Prediction the next word using trained model
prediction_index = tf.argmax(prediction, 1)

# Train the model over the number of iteration and the provided batch size
iteration = 50000
batchSize = 32
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  counter = 0
  for i in range(iteration):
      nextBatch = X[counter:counter+batchSize, :]
      nextBatch_label = Y[counter: counter+batchSize, :]
      # Report the accuracy every 5000 iteration
      if i % 5000 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: nextBatch, y_: nextBatch_label})
          print('step %d, training accuracy %g' % (i, train_accuracy))
          if train_accuracy > 0.99:
              print('reach the maximum accuracy, break out of iteration')
              break
      train_step.run(feed_dict={x: nextBatch, y_: nextBatch_label})
      counter += batchSize

      if counter + batchSize > len(X):
          nextBatch = X[counter:, :]
          nextBatch_label = Y[counter:, :]
          train_step.run(feed_dict={x: nextBatch, y_: nextBatch_label})
          counter = 0
  train_accuracy = accuracy.eval(feed_dict={x: nextBatch, y_: nextBatch_label})
  print('step %d, training accuracy %g' % (iteration, train_accuracy))
  # Generating text with the trained model
  # first: create a mapping dict from integer values to their corresponding words
  int_to_word = dict((int, word) for int, word in enumerate(vocab))
  # Randomly pick a sequence from the X data, and predict its next 50 words

  # pick a random seed
  start = np.random.randint(0, len(X_data) - 1)
  start_sequence = X_data[start]
  print("Random chosen sequence from the training set:")
  print("\"", ' '.join([int_to_word[word] for word in start_sequence]), "\"")

  # generate next 50 words
  next_50_words = []
  for i in range(50):
      X = np.reshape(start_sequence, (1, len(start_sequence), 1)) / float(len(vocab) - 1)
      index = prediction_index.eval(feed_dict={x: X})
      next_word = int_to_word[index[0]]
      next_50_words.append(next_word)
      start_sequence = start_sequence[1:] + [index[0]]

  print("Next 50 words: \n", ' '.join(next_50_words))



