import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import re


# Load data and prepare the input layers.
with open('test.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_sentence = []
    test_label = []
    for row in reader:
        test_sentence.append(row[1])
        if row[0] == 'postive':
            test_label.append([1, 0])
        else:
            test_label.append([0, 1])

test_label = np.array(test_label, dtype=int)

with open('train.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    train_sentence = []
    train_label = []
    for row in reader:
        train_sentence.append(row[1])
        if row[0] == 'postive':
            train_label.append([1, 0])
        else:
            train_label.append([0, 1])

train_label = np.array(train_label, dtype=int)

# Although it seems the sentences are clean in trained and test data, I applied a cleaning process to remove
# any punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
# All words also are converted to a lower case.
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")


def clean(sentence):
    return re.sub(remove_special_chars, "", sentence.lower())

# Create a data frame for wordVectors using the word-vectors.txt file for both train and test sets.
# Embedding matrix: the word vectors of all words in a sentences over all sentences.
wordVectors = pd.read_csv('word-vectors.txt', index_col=0, header=None)
max_words = 100
word_dim = 50
train_embedding_matrix = np.zeros((len(train_sentence), max_words, word_dim), dtype=np.float32)

for i in range(len(train_sentence)):
    indexCounter = 0
    clean_sentence = clean(train_sentence[i])
    split_sentence = clean_sentence.split()
    for word in split_sentence:
        try:
            train_embedding_matrix[i, indexCounter, :] = wordVectors.loc[word]
        except KeyError:
            pass
        indexCounter = indexCounter + 1
        if indexCounter >= max_words:
            break

test_embedding_matrix = np.zeros((len(test_sentence), max_words, word_dim), dtype=np.float32)
for i in range(len(test_sentence)):
    indexCounter = 0
    clean_sentence = clean(test_sentence[i])
    split_sentence = clean_sentence.split()
    for word in split_sentence:
        try:
            test_embedding_matrix[i, indexCounter, :] = wordVectors.loc[word]
        except KeyError:
            pass
        indexCounter = indexCounter + 1
        if indexCounter >= max_words:
            break

# Creating the vanilla RNN, LSTM, GRU models
# Specifying hyper-parameters
batchSize = 100
Units = 64  # number of units in RNNs
numClasses = 2
epoch = 1

# Input variables (place holders), wordVectors of a sentence (input x) and y_ (labels)
x = tf.placeholder(dtype=tf.float32, shape=[None, max_words, word_dim])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, numClasses])


RNNCell = tf.contrib.rnn.BasicLSTMCell(Units)
RNNCell = tf.nn.rnn_cell.BasicRNNCell(Units)
RNNCell = tf.contrib.rnn.GRUCell(Units)
RNNCell = tf.contrib.rnn.DropoutWrapper(cell=RNNCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(RNNCell, x, dtype=tf.float32)

# output layer for classification task
weight = tf.Variable(tf.truncated_normal([Units, numClasses]))
# weight = tf.get_variable('w_output', shape=[lstmUnits, numClasses])
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# bias = tf.get_variable('bias', shape=[numClasses], initializer=tf.constant_initializer(0.1))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


# Training model
# Use Adam as the optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Evaluate model
# Accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

''''
import datetime
tf.summary.scalar('Loss', cross_entropy)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
'''
# Train the model over number epochs and batch size
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for k in range(epoch):
      counter = 0
      for i in range(int(len(train_sentence)/batchSize)):
          nextBatch = train_embedding_matrix[counter:counter+batchSize, :]
          nextBatch_label = train_label[counter: counter+batchSize, :]
          train_step.run(feed_dict={x: nextBatch, y_: nextBatch_label})
          counter += batchSize
          if counter % 5000:
              train_accuracy = accuracy.eval(feed_dict={x: nextBatch, y_: nextBatch_label})
              print('After training %d examples, training accuracy %g' % (counter, train_accuracy))

  print('Test accuracy %g' % accuracy.eval(feed_dict={
      x: test_embedding_matrix, y_: test_label}))

