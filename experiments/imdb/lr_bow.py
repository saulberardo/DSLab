# -*- coding: utf-8 -*-
"""
Classify IMDB with Logistic Regression using BoW. 

# Accuracy
  0.85144 (with 5k words)
  0.87032 (with the entire vocab)

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config
from datasets.imdb import Imdb

# Load dataset
print('Loading dataset...')
imdb = Imdb(config.DATASETS_FOLDER)
(train_x_bow, train_y), (test_x_bow, test_y) = imdb.get_bow_and_categories(max_features=5000)


# Train LR model
print('Training model...')
lm = LogisticRegression()
lm.fit(train_x_bow, train_y)

# Predict and score on test set
ps = lm.predict(test_x_bow)
acc = accuracy_score(test_y, ps)
print(f'Accuracy: {acc}')



