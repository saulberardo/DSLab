# -*- coding: utf-8 -*-
"""
Load IMDB Dataset from FastAI repository (with little pre-processing) and train
it with Logistic Regression.

# More info: 
  hhttps://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset

# Pre-processing:
 - BoW

# Accuracy
  0.87032

"""

import pickle
import numpy as np
from keras.utils.data_utils import get_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Download full IMDB dataset
path = get_file('imdb_full.pkl',
                 origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                 md5_hash='d091312047c43cf9e4e38fef92437263')

# Load dataset
(train_x, train_y), (test_x, test_y) = pickle.load(open(path, 'rb'))


# Convert lists of IDs to strigs (so we can create BoW representations with sklearn)
train_x_str = [str(s).replace(',','').replace('[','').replace(']','') for s in train_x]
test_x_str = [str(s).replace(',','').replace('[','').replace(']','') for s in test_x]

# Vraete vectorizer to convert texts to BoW
cv = CountVectorizer()
cv.fit(train_x_str)

# Convert texts to BoW
train_x_bow = cv.transform(np.array(train_x_str))
test_x_bow = cv.transform(np.array(test_x_str))

# Train LR model
lm = LogisticRegression()
lm.fit(train_x_bow, train_y)

# Predict and score on test set
ps = lm.predict(test_x_bow)
acc = accuracy_score(test_y, ps)
print(acc)



