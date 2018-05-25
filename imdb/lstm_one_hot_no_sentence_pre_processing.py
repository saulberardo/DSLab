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
from tqdm import tqdm


NUM_WORDS = 1000

num_epochs = 2

NUM_SAMPLES = 1000

input_size = NUM_WORDS
n_hidden = 100
n_out = 2


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

lr = 0.3


train_accs = []
train_losses = []

print('Converting train sentences to OneHot')

# For ech apoch
for i_epoch in range(0, num_epochs):
    
    print(f'# Epoch {i_epoch}', flush=True)
          
        
    # Clasification probs for each sentences
    batch_probs = []
    
    batch_preds = []
    
    train_loss = 0
    
    # for each sentence in training set
    for j, x in enumerate(tqdm(train_x[0:NUM_SAMPLES])):

        #print(f'Sentence {j}/{NUM_SAMPLES}')
        sentence = T(word_idx_sequence_to_onehot_sequence(x))

        
        out, (h_t, c_t) = lstm(sentence.view(len(sentence), 1, -1))
        
        probs = softmax(linear(h_t.squeeze()))
        #batch_probs.append(probs.tolist())
        
        train_loss += loss(probs.view([1,2]), train_y[j:j+1])
        
        _, train_predicted_idx = probs.max(0)
        
        batch_preds.append(train_predicted_idx)
        
        
    train_loss = train_loss / NUM_SAMPLES        
        
    train_losses.append(train_loss)
    
    lstm.zero_grad()
    linear.zero_grad()
    
    train_loss.backward()
    

    
    # Update parameters
    for param in lstm.parameters():
        param.data -= lr * param.grad.data
        
    # Update parameters
    for param in linear.parameters():
        param.data -= lr * param.grad.data
    
    #train_loss = loss(T(batch_probs), train_y[0:100])
    #print(train_loss)
    
    # Compute train set accuracy 
    train_acc = accuracy_score(train_y[0:NUM_SAMPLES].data.numpy(), batch_preds)
    train_accs.append(train_acc)
    train_losses.append(train_loss.item())
    
    print(train_loss.item(), train_acc)    
    

    
plt.plot(train_losses, label='Train Loss', color='blue')    
plt.show()
    
    
    
    
    
    
    
    
    
    

