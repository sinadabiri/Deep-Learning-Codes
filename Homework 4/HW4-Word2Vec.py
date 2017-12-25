# Problem 3
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
%matplotlib inline

fh = open("vectors_new.txt", "r", encoding='utf-8')
word2vec = {}
all_words = []
all_word2vectors = []
for line in fh:
    split = line.split()
    word = split[0]
    values = [float(value) for value in split[1:]]
    all_words.append(word)
    all_word2vectors.append(values)
    word2vec.update({word: values})

subject_words = ['life', 'market', 'stanford', 'trump', 'public']

# Q-1 Semantics
similar_words_for_all_words = []  # list of lists where each sublist contains similar words to a subject word.
for word in subject_words:
    similarity = {}
    for key in word2vec.keys():
        similarity.update({key: 1-cosine(word2vec[word], word2vec[key])})
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

    most_similar_words = []
    for i in range(1, 21):  # Since the first similar word is the word itself, I discard it.
        most_similar_words.append(sorted_similarity[i][0])
    similar_words_for_all_words.append(most_similar_words)
    print('The 20 most similar words to the word: {} are: {}'.format(word, most_similar_words))
    print('\n')
# ===================================================================================================
# Q-2 Visualization
# To reduce the computation time, I just used 250 iteration in TSNE.
# part (a): tsne for all words.
# In terms of word names in the plot, I just annotated the first 100 word names since showing all number of words
# makes the figure screen totally black.
tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=30, n_iter=250)
embedded_words = tsne.fit_transform(np.array(all_word2vectors))
plt.figure(1)
ax = plt.gca()
ax.scatter(list(embedded_words[:, 0]), list(embedded_words[:, 1]), s=10)
ax.set_title('t-sne for all words in 2-dimensional')
for i, word in enumerate(all_words[:100]):
    ax.annotate(word, (embedded_words[i, 0], embedded_words[i, 1]))

# ===================================================================================================
# Part (b)
# Visualization plots of the most 20 similar words to each subjected words.
for i, word in enumerate(subject_words):
    plt.figure(i+2)
    ax = plt.gca()
    embedded_words_20 = np.zeros((20, 2))
    for j, item in enumerate(similar_words_for_all_words[i]):
        embedded_words_20[j, :] = embedded_words[all_words.index(item), :]
    ax.scatter(list(embedded_words_20[:, 0]), list(embedded_words_20[:, 1]))
    ax.set_title("The 20 most similar words to the word: %s" % word)
    for k, item1 in enumerate(similar_words_for_all_words[i]):
        ax.annotate(item1, (embedded_words_20[k, 0], embedded_words_20[k, 1]))
