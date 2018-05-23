# -*- coding: utf-8 -*-
"""
Load IMDB Dataset from Keras and train with LogisticRegression. There is also a
function in the end to return the original text of a review.

# More info: 
  https://keras.io/datasets/

# Pre-processing:
 - Remove infrequent words
 - BoW

# Accuracy
 0.86432 (with 1k words)
 0.85904 (with 10k words)


"""
import keras
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


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


# Convert lists of IDs to strigs (so we can create BoW representations with sklearn)
train_x_str = [str(s).replace(',','').replace('[','').replace(']','') for s in train_x]
test_x_str = [str(s).replace(',','').replace('[','').replace(']','') for s in test_x]

# Vraete vectorizer to convert texts to BoW
cv = CountVectorizer()
cv.fit(train_x_str)

# Convert texts to BoW
train_x_bow = V(T(cv.transform(np.array(train_x_str)).toarray()), requires_grad=False)
test_x_bow = V(T(cv.transform(np.array(test_x_str)).toarray()), requires_grad=False)

train_y = V(T(train_y), requires_grad=False).long()
test_y = V(T(test_y), requires_grad=False).long()

train_mean = train_x_bow.mean(0)
train_std = train_x_bow.std(0)

train_x_bow = (train_x_bow - train_mean) / train_std
test_x_bow = (test_x_bow - train_mean) / train_std

n_in = train_x_bow.shape[1]
n_hidden = 100
n_out = 2

model = nn.Sequential(
  nn.Linear(n_in, n_hidden),
  nn.ReLU(),
  nn.Dropout(p=0.95),
  nn.Linear(n_hidden, n_hidden),
  nn.ReLU(),
  nn.Dropout(p=0.95),
  nn.Linear(n_hidden, n_out),
  nn.Softmax()
)

loss = CrossEntropyLoss()





train_accs = []
test_accs = []
train_losses = []
test_losses = []

lr = 3.
num_epochs = 50
for i_epoch in range(0, num_epochs):
    
    probs = model(train_x_bow)
    
    train_loss = loss(probs, train_y)
    
    # Zero grad then copute grads
    model.zero_grad()
    train_loss.backward()
    
    # Update parameters
    for param in model.parameters():
        param.data -= lr * param.grad.data
        
    # Compute train set accuracy 
    _, train_predicted_idx = probs.max(1)
    train_acc = accuracy_score(train_y.data.numpy(), train_predicted_idx.data.numpy())
    train_accs.append(1 - train_acc)
    train_losses.append(train_loss.data[0])
    
    # Compute teste set loss and accuracy   
    test_probs = model(test_x_bow)
    test_loss = loss(test_probs, test_y)
    test_losses.append(test_loss.data[0])
    
    
    _, test_predicted_idx = test_probs.max(1)
    test_acc = accuracy_score(test_y.data.numpy(), test_predicted_idx.data.numpy())
    test_accs.append(1 - test_acc)
        
    print(f'{i_epoch}: Train: [{train_acc}, {train_loss.data[0]}], Test: [{test_acc}, {test_loss.data[0]}]')    
    
    

    
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.legend(loc='upper center')
plt.twinx()
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.legend()
plt.show()
