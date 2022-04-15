# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
from scipy.linalg import svd
from numpy import dot




def get_vocabulary(file_path):
    vocab = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            review_text = line.split()
            for word in review_text:
                if word not in vocab.keys():
                    vocab[word] = i
                    i += 1
    return vocab


vocab = get_vocabulary('data/reviews.txt')

matrix = np.zeros((len(vocab), len(vocab)))



def make_cooccurrence_matrix(file_path, vocab):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            review_text = line.split()
            for i in range(len(review_text)-1):
                matrix [vocab[review_text[i]], vocab[review_text[i+1]]] += 1
                matrix [vocab[review_text[i+1]], vocab[review_text[i]]] += 1
    return matrix

com = make_cooccurrence_matrix('data/reviews.txt', vocab)

# store the matrix in a new file called cooccurrence_matrix.txt
with open('data/cooccurrence_matrix.txt', 'w') as f:
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            f.write(str(com[i,j]) + " ")
        f.write('\n')


U, S, V = np.linalg.svd(com)

word = {}

index = 0 
k = 10
for i in vocab.keys():
    word[i] = U[index][:k]
    index += 1

print(word)


