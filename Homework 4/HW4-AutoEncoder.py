import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import pickle
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Building the training dataset the same as requested in the homework.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch = mnist.train.next_batch(100)
images = mnist.train.images
labels = mnist.train.labels
training_images = [[] for _ in range(10)]
training_labels = [[] for _ in range(10)]
counter = np.zeros((10, 1))
for i in range(len(images)):
    if min(counter) > 100:
        break
    digit = int(list(labels[i]).index(1))
    training_images[digit].append(images[i])
    training_labels[digit].append(labels[i])
    counter[digit, 0] = counter[digit, 0] + 1
for i in range(10):
    training_images[i] = training_images[i][:100]
    training_labels[i] = training_labels[i][:100]
del mnist

training_images = [image for set in training_images for image in set]
training_images = np.array(training_images)
training_labels = [label for set in training_labels for label in set]
training_labels = np.array(training_labels)

with open('images_labels.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([training_images, training_labels], f)
# Building the encoder
A = 2
def encoder(x, num_hidden):
    # Encoder Hidden layer with sigmoid activation
    encoder_h = tf.Variable(tf.random_normal([num_input, num_hidden]))
    encoder_b = tf.Variable(tf.random_normal([num_hidden]))
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h), encoder_b))
    return layer_1


# Building the decoder
def decoder(layer_1, num_hidden):
    # Decoder Hidden layer with sigmoid activation
    decoder_h = tf.Variable(tf.random_normal([num_hidden, num_input]))
    decoder_b = tf.Variable(tf.random_normal([num_input]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, decoder_h), decoder_b))
    return layer_2


# Network Parameters
num_hidden_list = [2, 5, 10] # num of hidden units, chosen from [2, 5, 10]
num_input = 784  # MNIST data input (img shape: 28*28)

# Training Parameters
num_steps = 200
batch_size = 32
display_step = 100
# Part (a) for AutoEncoder, Reconstruction error for each of the hidden unit per every 100 training iterations
error_AE = [[], [], []]

# Part (b) for AutoEncoder, label_prediction and true_label for every value of N.
true_pred_labels_AE = [[], [], []]

for k in range(len(num_hidden_list)):
    X = tf.placeholder(tf.float32, shape=[None, num_input])

    # Construct model
    encoder_part = encoder(X, num_hidden=num_hidden_list[k])
    decoder_part = decoder(encoder_part, num_hidden=num_hidden_list[k])

    # Prediction
    y_pred = decoder_part
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error, and train_step
    # Loss or reconstruction error is the average of squared euclidean distance for samples in batch-size
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y_true - y_pred, 2), 1))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Start Training

    with tf.Session() as sess:
        # Run the initializer
        sess.run(tf.global_variables_initializer())

        # Training
        counter = 0
        for i in range(1, num_steps+1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = training_images[counter: counter+batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)
            optimizer.run(feed_dict={X: batch_x})
            loss_value = loss.eval(feed_dict={X: batch_x})
            # _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            # Display logs per step
            if i % display_step == 0:
                print('Step %i: Average of Minibatch Reconstruction Error: %f' % (i, loss_value))
                error_AE[k].append(loss_value)

            counter += batch_size
            if (counter + batch_size) > 1000:
                # Run optimization and get loss value)
                batch_x = training_images[counter:]
                optimizer.run(feed_dict={X: batch_x})
                loss_value = loss.eval(feed_dict={X: batch_x})
                counter = 0

        # Part (b): For each N, compute the predicted label for 50 nearest images to the first image.
        # Use cosine similarity for finding the nearest images.
        latent_space_AE = encoder_part.eval(feed_dict={X: training_images})
        one_image = latent_space_AE[0]
        label_pred = [int(list(training_labels[0]).index(1))] * 50
        distance = []
        for i in range(1, len(latent_space_AE)):
            distance.append((i, 1 - cosine(latent_space_AE[i], one_image)))
        sorted_distance = sorted(distance, key=lambda x: x[1], reverse=True)
        nearest_images = sorted_distance[:50]
        label_true = [int(list(training_labels[item[0]]).index(1)) for item in nearest_images]
        true_pred_labels_AE[k].extend([label_true, label_pred])

# Part (a)for AutoEncoder
# Plot the reconstruction error. For each value N, the reconstruction error over iterations has been shown.
# Note that the results has been obtained for every 100 iteration.
plt.figure(1)
for i in range(3):
    iteration = np.linspace(100, num_steps, int(num_steps/100))
    label = 'N = ' + str(num_hidden_list[i])
    plt.plot(iteration, error_AE[i], label=label)
    plt.ylabel("Average squared euclidean distance")
    plt.xlabel("Number of iteration")
    plt.title('Average Reconstruction Error Versus Iteration for Different Hidden Units')
plt.legend(loc='upper right')
plt.show()

# Part (b) for AutoEncoder
# Reporting precision and recall corresponding to each digit per each value N.
for i, N in enumerate(num_hidden_list):
    print('Precision, recall, F-score for each digit for N = {}'.format(N))
    print(classification_report(true_pred_labels_AE[i][0], true_pred_labels_AE[i][1]))

# Part C, PCI
error_pci = []
true_pred_labels_pci = [[], [], []]
for i in range(len(num_hidden_list)):
    pca = PCA(n_components=num_hidden_list[i])
    latent_space_pca = pca.fit_transform(training_images)
    # Part (a) for PCI,
    # Having the original and transform original, we can compute the reconstruction error.
    transform_original = pca.inverse_transform(latent_space_pca)
    error_pci.append(np.mean(np.sum(np.power(training_images - transform_original, 2), axis=1)))

    # Part (b) for PCI:
    # For each N, compute the predicted label for 50 nearest images to the first image.
    # Use cosine similarity for finding the nearest images.
    one_image = latent_space_pca[0]
    label_pred = [int(list(training_labels[0]).index(1))] * 50
    distance = []
    for j in range(1, len(latent_space_pca)):
        distance.append((j, 1 - cosine(latent_space_pca[j], one_image)))
    sorted_distance = sorted(distance, key=lambda x: x[1], reverse=True)
    nearest_images = sorted_distance[:50]
    label_true = [int(list(training_labels[item[0]]).index(1)) for item in nearest_images]
    true_pred_labels_pci[i].extend([label_true, label_pred])

# Report Results for part c
# Comparing the AutoEncoder and PCI Reconstruction error:
for i, N in enumerate(num_hidden_list):
    print('Reconstruction error for N = {}: Autoencoder = {}, PCI = {}'.format(N, min(error_AE[i]), error_pci[i]))
# Part (b) for PCI
# Reporting precision and recall corresponding to each digit per each value N.
for i, N in enumerate(num_hidden_list):
    print('Precision, recall, F-score for each digit for N = {}'.format(N))
    print(classification_report(true_pred_labels_pci[i][0], true_pred_labels_pci[i][1]))