### Assignment 2 - NLP

Report 

#### 2.1 Theory

Negative Sampling minimises the similarity of the word if they have different context and maximises the similarity when they occur in different contexts.

Instead of minimising all words in the dictionary except for context words, it randomly chooses some words based on the training size utilizes them to optimise the goal. We use a bigger k for smaller dataset and vice-versa, the computation is done over k negative samples from the noise distribution, which is easily doable as opposed to computing the softmax over the entire dictionary.

### 2.2 Implementation

Part 1: First Run the Assignment.py file to get the cleaned data in a text file then run model1.py file to get the word embeddings.
Part 2: Again run the assignment.py to get amount of cleaned data that you desire then run model2.py to get the closest words to a given words as well as word embeddings.

### 2.3 Analysis 

1. For the 2nd Model the output is stored in a file named "2-3-i-Model2.png"
2. For the 2nd Model the output is stored in a file named "2-3-ii-M2.txt", For the first Model it is stored in the file "2-3-ii-M1.txt"


###### Model 1 
I have taken the first 30000 sentences from the dataset, then stored the "reviewText" attribute of the in a text file. Then made a co-occurance matrix of the same, and then used scipy's svd function to get the singular values and vectors of the matrix and using them computed the word embeddings.

##### Model 2
I have used ginsim's w2v function to get the the vectors and then trained the vectors using CBOW model with negative-sampling, Later in the same file is there is the implementation of t-SNE to get a 2-D plot of the words closest to a given word.

##### Co-occurance Matrix
I have generated the Co-occurance matrix of the first 30000 sentences from the dataset using two methods first by using a library and the output of that is stored in file co-occurance.csv and then by using then without using the library and the output is stored in file data/co-occurance-matrix.csv