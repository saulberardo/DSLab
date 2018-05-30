# -*- coding: utf-8 -*-
"""
Classify IMDB with Neural Network using BoF.

# Accuracy
  A simple two layer NN can achieve easily above 0.83

# Comments
   * If I train with dropout, after finishing training, if I train for one more 
 epoch without it, the accuracy increases to above 0.84.
"""

from torch import FloatTensor as T
from torch.autograd import Variable as V
from torch.nn import CrossEntropyLoss
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import config
from datasets.imdb import Imdb


# Load dataset
print('Loading dataset...', flush=True)
imdb = Imdb(config.DATASETS_FOLDER)
(train_x_bof, train_y), (test_x_bof, test_y) = imdb.get_bof_fasttext_wiki_news_300d_1M()


# Pack dataset to torch Variables
train_x_bof = V(T(train_x_bof), requires_grad=False)
test_x_bof = V(T(test_x_bof), requires_grad=False)
train_y = V(T(train_y), requires_grad=False).long()
test_y = V(T(test_y), requires_grad=False).long()

# Compute train mean and std
train_mean = train_x_bof.mean(0)
train_std = train_x_bof.std(0)

# Normalize train and test sets
train_x_bof = (train_x_bof - train_mean) / train_std
test_x_bof = (test_x_bof - train_mean) / train_std

# Network parameters
n_in = train_x_bof.shape[1]
n_hidden = 100
n_out = 2
loss = CrossEntropyLoss()

# Network model
model = nn.Sequential(
  nn.Linear(n_in, n_hidden),
  nn.ReLU(),
  nn.Dropout(p=0.5),
  nn.Linear(n_hidden, n_hidden),
  nn.ReLU(),
  nn.Dropout(p=0.5),
  nn.Linear(n_hidden, n_out),
  nn.Softmax()
)


# List to store training statistics
train_accs = []
test_accs = []
train_losses = []
test_losses = []

# Training parameters
lr = .1
num_epochs = 90

# Train model
print('Training model...')
for i_epoch in range(0, num_epochs):
    
    # Foward pass
    probs = model(train_x_bof)
    
    # Compute loss
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
    
    # Compute test set loss and accuracy   
    test_probs = model(test_x_bof)
    test_loss = loss(test_probs, test_y)
    test_losses.append(test_loss.data[0])
    
    # Compute test set loss and accuracy
    _, test_predicted_idx = test_probs.max(1)
    test_acc = accuracy_score(test_y.data.numpy(), test_predicted_idx.data.numpy())
    test_accs.append(1 - test_acc)
        
    print(f'{i_epoch}: Train: [{train_acc}, {train_loss.data[0]}], Test: [{test_acc}, {test_loss.data[0]}]')    
    
    
# Plot statistics
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.legend(loc='upper center')
plt.twinx()
plt.plot(train_accs, label='Train Error')
plt.plot(test_accs, label='Test Error')
plt.legend()
plt.show()

import torch
def update_dropout(model, p):
    for m in model.modules():
        if type(m) is torch.nn.modules.dropout.Dropout:
            m.p = p