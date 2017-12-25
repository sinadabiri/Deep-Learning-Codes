import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Sina\Desktop\Deep Learning\Assign1')
np.random.seed(10)

trainData = pd.read_csv('train_data.txt', header=0, names=['x1', 'x2', 'class'])
testData = pd.read_csv('test_data.txt', header=0, names=['x1', 'x2', 'class'])

# X_test: the input matrix used in test process
X_test = np.transpose(np.array(testData.loc[:, :'x2']))

# y_test_hat: the one-hot target vector
y_test_hat = np.array(testData.loc[:, 'class'])


# X: the input matrix used in training process
X = np.transpose(np.array(trainData.loc[:, :'x2']))

# y_hat: the one-hot target vector for training process
y_hat = np.transpose(np.array(trainData.loc[:, 'class']))
zeros = np.zeros((2, len(y_hat)))
for i in range(len(y_hat)):
    zeros[y_hat[i], i] = 1
y_hat = zeros

# W_i: Matrix of weights at each layer. No. rows = no. neurons in next layer. No. cols = no. neurons in current layer.
# First, we randomly initialize them between [0, 1)
W_3 = np.random.rand(2, 3)
W_2 = np.random.rand(3, 3)
W_1 = np.random.rand(3, 2)

# a_l_sigmoid: vector of a^l at layer 1 and 2, which is a^l = sigmoid(W^1.X)
def a_l_sigmoid(W, X):
    z = np.matmul(W, X)
    exponential = np.exp(-z)
    shape = np.shape(exponential)
    return np.divide(np.ones(shape),(np.ones(shape) + exponential))

# a_3_softmax: vector of y at the last layer (i.e., a^3)
def a_3_softmax(W, a):
    z = np.matmul(W_3, a)
    exponential = np.exp(z)
    Sum = sum(exponential)
    return exponential/Sum

# We assume the eta as 0.01
eta = 0.01

# Batch size is considered as 1 since the question asks to apply SGD without saying anything about batch_size
# We apply SDG with batch size(bs) = 1, and number of epoc = 1
Number_epoc = 1

for epoc in range(Number_epoc):
    for bs in range(np.shape(X)[1]):
        X_ = np.reshape(X[:, bs], (2, 1))
        y_hat_ = np.reshape(y_hat[:, bs], (2, 1))

        # Compute a_l and z_l at different layers
        z_1 = np.matmul(W_1, X_)
        a_1 = a_l_sigmoid(W_1, X_)

        z_2 = np.matmul(W_2, a_1)
        a_2 = a_l_sigmoid(W_2, a_1)

        z_3 = np.matmul(W_3, a_2)
        a_3 = a_3_softmax(W_3, a_2)

        # Print loss function in this step
        loss = sum(np.power(y_hat_ - a_3, 2))
        print("Value of loss function at setp {0} is: {1}".format(bs, loss))

        # dev_firstpart_l: The first part of derivative in layer l ==> partial dev (C) w.r.t w_ij
        dev_firstpart_3 = np.zeros((2, 3), dtype=np.float32)
        dev_firstpart_2 = np.zeros((3, 3), dtype=np.float32)
        dev_firstpart_1 = np.zeros((3, 2), dtype=np.float32)

        for i in range(2):
            dev_firstpart_3[i, :] = np.transpose(a_2)
        for i in range(3):
            dev_firstpart_2[i, :] = np.transpose(a_1)
        for i in range(3):
            dev_firstpart_1[i, :] = np.transpose(X_)

        # dev_secondpart_l: The second part of derivative in layer l ==> partial dev (C) w.r.t z_i^l == delta_l

        # delta_3 for layer 3:
        exp_z_3 = np.exp(z_3)
        Sum = sum(exp_z_3)
        sigma_prime_3 = np.zeros((2, 1))
        for i in range(2):
            sigma_prime_3[i, 0] = (exp_z_3[i, 0] * Sum - exp_z_3[i, 0]**2)/(Sum**2)
        delta_3 = np.multiply(sigma_prime_3, 2*(a_3 - y_hat_))

        # delta_2 for layer 2:
        sigma_prime_2 = np.multiply(z_2, np.ones(np.shape(z_2)) - z_2)
        delta_2 = np.multiply(sigma_prime_2, np.matmul(np.transpose(W_3), delta_3))

        # delta_1 for layer 1:
        sigma_prime_1 = np.multiply(z_1, np.ones(np.shape(z_1)) - z_1)
        delta_1 = np.multiply(sigma_prime_1, np.matmul(np.transpose(W_2), delta_2))

        # For each layer, I first expand the delta vector to a matrix by duplicating columns.
        # This is for the purpose of element wise multiplication
        delta_3_matrix = np.zeros((2, 3))
        for i in range(3):
            delta_3_matrix[:, i] = delta_3[:, 0]

        delta_2_matrix = np.zeros((3, 3))
        for i in range(3):
            delta_2_matrix[:, i] = delta_2[:, 0]

        delta_1_matrix = np.zeros((3, 2))
        for i in range(2):
            delta_1_matrix[:, i] = delta_1[:, 0]

        # Now we compute the Gradient at each layer Gradient_l: element wise multiplication of first and second parts

        Gradient_3 = np.multiply(dev_firstpart_3, delta_3_matrix)
        Gradient_2 = np.multiply(dev_firstpart_2, delta_2_matrix)
        Gradient_1 = np.multiply(dev_firstpart_1, delta_1_matrix)

        # Update parameters (W_l) at each layer using SGD

        W_3 = W_3 - eta * Gradient_3
        W_2 = W_2 - eta * Gradient_2
        W_1 = W_1 - eta * Gradient_1

# precision, recall, fscore
# First predict the test data using trained network (Updated W in the last iteration)

Pred = np.zeros((2, 1000))
for i in range(1000):
    X_test_ = np.reshape(X[:, i], (2, 1))
    a_1 = a_l_sigmoid(W_1, X_test_)
    a_2 = a_l_sigmoid(W_2, a_1)
    a_3 = a_3_softmax(W_3, a_2)
    Pred[:, i] = a_3[:, 0]

Pred_Label = np.argmax(Pred, axis=0)

# Considering label 1 = positive, label 0 = negative
# Find TP, TN, FN, FP, and then compute precision, recall, and Fscore accordingly.

Index_Actual_Positive = np.where(y_test_hat == 1)[0]
Index_Actual_Negative = np.where(y_test_hat == 0)[0]

True_Positive = 0
True_Negative = 0

for i in Index_Actual_Positive:
    if Pred_Label[i] == 1:
        True_Positive += 1
False_Negative = len(Index_Actual_Positive) - True_Positive

for i in Index_Actual_Negative:
    if Pred_Label[i] == 0:
        True_Negative += 1
False_Positive = len(Index_Actual_Negative) - True_Negative

Precision = True_Positive/(True_Positive + False_Positive)
Recall = True_Positive/(True_Positive + False_Negative)
F_score = 2/(1/Precision + 1/Recall)

print('Performance measure of model in question 1 with one epoch and batch size = 1: ', '\n')
print("Precision {0}, Recall {1}, F_score {2}".format(Precision, Recall, F_score))