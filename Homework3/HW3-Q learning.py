import csv
import numpy as np
with open('q-learning.dat', 'r', newline='') as f:
    reader = csv.reader(f)
    train_data = []
    for row in reader:
        row = row[0].strip().split()
        train_data.append(row)

train_data = np.array(train_data, dtype=float)

gama = 0.9
alfa = 0.1
Q = np.zeros((2, 2))

initial_s = np.random.randint(1, 3, 1)
convergence = 0.2
threshold = 0.01

iteration = 0
while threshold < convergence:
    Q_new = np.zeros((2, 2))
    iteration += 1
    for i in range(len(train_data)):
        s_prime = train_data[i, 2]
        target = train_data[i, 3] + gama * max(Q[int(s_prime) - 1, :])
        Q_new[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] = (1 - alfa) * Q[int(train_data[i, 0]) - 1,
                                                                                     int(train_data[i, 1]) - 1] + alfa * target
    if iteration == 1:
        print('Q(s, a) values after going through data once: ', Q_new)

    convergence = np.amax(Q_new - Q)
    Q = Q_new

print('Convergence value {} and Iteration number {}'.format(convergence, iteration))
print('Q(s, a) values at convergence: ', Q, '\n\n')

# Part 3
gama = 0.9
Q = np.zeros((2, 2))
convergence = 0.2
threshold = 0.01

iteration = 0
w = np.zeros((2, 2))
while threshold < convergence:
    Q_new = np.zeros((2, 2))
    iteration += 1
    for i in range(len(train_data)):
        s_prime = train_data[i, 2]
        target = train_data[i, 3] + gama * max(Q[int(s_prime) - 1, :])
        alfa = 1./(w[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] + 1)
        Q_new[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] = (1 - alfa) * Q[int(train_data[i, 0]) - 1,
                                                                                     int(train_data[i, 1]) - 1] + alfa * target
        w[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] += 1
    if iteration == 1:
        print('alfa = 1/(w+1): Q(s, a) values after going through data once: ', Q_new)

    convergence = np.amax(Q_new - Q)
    Q = Q_new

print('alfa = 1/(w+1): Convergence value: {}, and Iteration number: {}'.format(convergence, iteration))
print('alfa = 1/(w+1): Q(s, a) values at convergence: ', Q)