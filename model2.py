import imp
import numpy as np
import pandas as pd
import json
import re
import nltk
import gensim.downloader as api

nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


file_path = 'data/reviews.json'

no_of_lines = 30000

# Selecting the first no_of_lines reviews


# def select_lines(file_path, no_of_lines):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         lines = lines[:no_of_lines]
#         with open('data/header.json', 'w') as f:
#             f.writelines(lines)


# Storing the review attributes of the json in a json file
def store_review(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            review = json.loads(line)
            review_attributes = {'reviewText': review['reviewText']}
            with open('../reviews.json', 'a') as f:
                f.write(json.dumps(review_attributes) + '\n')


# store the review text in a txt file
# def store_review_text(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             review = json.loads(line)
#             with open('../reviews.txt', 'a') as f:
#                 f.write(review['reviewText'] + '\n')

vocab_size = 0
embedding_size = 10

# Opening JSON file
store_review("data/reviews.json")
f = open('data/reviews.json', )

records = [json.dumps(line) for line in f]
final_records = [json.loads(lines) for lines in records]
print(len(final_records))
# print (final_records[0])
all_words = [nltk.word_tokenize(final_records[i]) for i in range(len(final_records))]


word2vec = Word2Vec(all_words, sg = 0, hs = 0, vector_size=25)
# vocabulary = word2vec.wv.vocab
# print(vocabulary)

# v1 = word2vec.wv['artificial']
# print(len(v1))

sim_words = word2vec.wv.most_similar('camera' , topn=10)
print(sim_words)

# wv = api.load('word2vec-google-news-300')
# print(wv.most_similar("camera", topn=10))


keys = ['camera', 'product', 'good', 'strong', 'look']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in word2vec.wv.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(word2vec.wv[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

tsne_model_en_2d = TSNE(perplexity=10,
                        n_components=2,
                        init='pca',
                        n_iter=3500,
                        random_state=32)
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
embeddings_en_2d = np.array(
    tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m,
                                                              k))).reshape(
                                                                  n, m, 2)


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters,
                                               word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word,
                         alpha=0.5,
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("fг.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)