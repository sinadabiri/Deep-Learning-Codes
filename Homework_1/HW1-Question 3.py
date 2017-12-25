# Question 3 and Question 4
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense

os.chdir(r'C:\Users\Sina\Desktop\Deep Learning\Assign1')

trainData = np.array(pd.read_csv('train_data.txt', header=0, names=['x1', 'x2', 'class']))
testData = np.array(pd.read_csv('test_data.txt', header=0, names=['x1', 'x2', 'class']))

Train_X = trainData[:, :2]
Train_Y = np.reshape(trainData[:, 2], (1000, 1))
Train_Y = keras.utils.to_categorical(Train_Y, num_classes=2)

Test_X = testData[:, :2]
Test_Y_original = np.reshape(testData[:, 2], (1000, 1))
Test_Y = keras.utils.to_categorical(Test_Y_original, num_classes=2)

# Question 3
model = Sequential()
model.add(Dense(3, activation='sigmoid', input_dim=2))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(Train_X, Train_Y, epochs=5, batch_size=50)
# End of Question 3


# Question 4
# precision, recall, fscore
# First predict the test data using trained network
Pred = model.predict(Test_X, batch_size=1)
Pred_Label = np.argmax(Pred, axis=1)

# Considering label 1 = positive, label 0 = negative
# Find TP, TN, FN, FP, and then compute precision, recall, and Fscore accordingly.

Index_Actual_Positive = np.where(Test_Y_original == 1)[0]
Index_Actual_Negative = np.where(Test_Y_original == 0)[0]

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

print('\n')
print('Performance measure of model in question 2 with one epoch and batch size = 1: ', '\n')
print("Precision {0}, Recall {1}, F_score {2}".format(Precision, Recall, F_score))
