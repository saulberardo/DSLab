# -*- coding: utf-8 -*-
"""


"""
import keras
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

import torch
from torch import FloatTensor as T
from torch.autograd import Variable as V
from torch.nn import CrossEntropyLoss
from torch import nn
import matplotlib.pyplot as plt




# only use top 1000 words ( I don't know why, but it uses always 10 less words than specified here)
NUM_WORDS = 100

# Word indexes start at 3 (so we can set 0, 1, and 2 with special tags)
INDEX_FROM = 3   # word index offset

# Get train and test data
(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_y = V(T(train_y), requires_grad=False).long()
test_y = V(T(test_y), requires_grad=False).long()


def word_idx_sequence_to_onehot_sequence(word_idx_sequence):
    return [word_idx_to_onehot(word_idx) for word_idx in word_idx_sequence]

def word_idx_to_onehot(word_idx): 
    r=np.zeros(NUM_WORDS)
    r[word_idx] = 1
    return r



input_size = NUM_WORDS
n_hidden = 10
n_out = 2

#
lstm = nn.LSTM(NUM_WORDS, n_hidden)
linear = nn.Linear(in_features=n_hidden, out_features=n_out)
softmax = nn.Softmax()


"""
# For each word in the sentence
for word in sentence:    
    # Compute new output, hidden and cell states
    out, (h_t, c_t) = lstm(word.view(1, 1, -1), (h_t, c_t))
"""

"""
# Reshape the sequence 
sentence = sentence.view(len(sentence), 1, -1)
# Process the hole sequence in a single foward pass
out, (h_t, c_t) = lstm(sentence, (h_t, c_t))
"""

loss = CrossEntropyLoss()

lr = 3.
num_epochs = 5

# For ech apoch
for i_epoch in range(0, num_epochs):
    
    print(f'# Epoch {i_epoch}')
        
    # Clasification probs for each sentences
    batch_probs = []
    
    # for each sentence in training set
    for j, x in enumerate(train_x[0:100]):

        print(f'Sentence {j}/{len(train_x)}')
        
        #  
        sentence = T(word_idx_sequence_to_onehot_sequence(x))
        
        # initialize the hidden and cell states
        h_t = torch.zeros(1, 1, n_hidden).view(1,1,10)
        c_t = torch.zeros(1, 1, n_hidden).view(1,1,10)
        
        out, (h_t, c_t) = lstm(sentence.view(len(sentence), 1, -1), (h_t, c_t))
        
        probs = softmax(linear(h_t.squeeze()))
        batch_probs.append(probs.tolist())
        
        
        
        
    train_loss = loss(T(batch_probs), train_y[0:100])
    print(train_loss)


    
    
    
    
    
    
    
    
    
    
    
    
    

