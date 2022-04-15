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


file_path = 'data/Electronics_5.json'


no_of_lines = 350

# Selecting the first no_of_lines reviews


# write a function to delete a file
def delete_file(file_path):
    with open(file_path, 'w'):
        pass


delete_file('data/reviews.txt')
delete_file('data/tokens.txt')
delete_file('data/header.json')
delete_file('data/reviews.json')


def select_lines(file_path, no_of_lines):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[:no_of_lines]
        with open('data/header.json', 'w') as f:
            f.writelines(lines)


# Storing the review attributes of the json in a json file
def store_review(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            review = json.loads(line)
            review_attributes = {
                'reviewText': review['reviewText']}
            with open('data/reviews.json', 'a') as f:
                f.write(json.dumps(review_attributes) + '\n')


# store the review text in a txt file
text_data = []


def store_review_text(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            review = json.loads(line)
            with open('data/reviews.txt', 'a') as f:
                text_data.append(review['reviewText'])
                f.write(review['reviewText']+'\n')


select_lines('data/Electronics_5.json', no_of_lines)
store_review('data/header.json')
store_review_text('data/reviews.json')

text_lines = []

with open('data/reviews.txt', 'r') as f:
    text_lines = f.readlines()


# # tokenize each line in the text file
# def tokenize(text_lines):
#     tokenized_lines = []
#     for line in text_lines:
#         # tokenized_line = word_tokenize(line)
#         tokenized_line = [token.lower() for token in tokenized_line]
#         tokenized_lines.append(tokenized_line)
#         with open('data/tokens.txt', 'a') as f:
#                 f.write(str(tokenized_line)+'\n')
#     return tokenized_lines

# make a list of all words in file reviews.txt
# def make_list(tokenized_lines):
#     all_words = []
#     for line in tokenized_lines:
#         for word in line:
#             all_words.append(word)
#     return all_words


# tokenized_lines = tokenize(text_lines)

# def get_vocabulary(file_path):
#     vocab = {}
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         i = 0
#         for line in lines:
#             review_text = line.split()
#             for word in review_text:
#                 if word not in vocab.keys():
#                     vocab[word] = i
#                     i += 1
#     return vocab


# vocab = get_vocabulary('data/reviews.txt')

# matrix = np.zeros((len(vocab), len(vocab)))



# def make_cooccurrence_matrix(file_path, vocab):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             review_text = line.split()
#             for i in range(len(review_text)-1):
#                 matrix [vocab[review_text[i]], vocab[review_text[i+1]]] += 1
#                 matrix [vocab[review_text[i+1]], vocab[review_text[i]]] += 1
#     return matrix

# com = make_cooccurrence_matrix('data/reviews.txt', vocab)

# # store the matrix in a new file called cooccurrence_matrix.txt
# with open('data/cooccurrence_matrix.txt', 'w') as f:
#     for i in range(len(vocab)):
#         for j in range(len(vocab)):
#             f.write(str(com[i,j]) + " ")
#         f.write('\n')


# U, S, V = np.linalg.svd(com)

# word = {}

# index = 0 
# k = 10
# for i in vocab.keys():
#     word[i] = U[index][:k]
#     index += 1

# print(word)

