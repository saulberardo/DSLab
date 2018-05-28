# -*- coding: utf-8 -*-
"""
Classify IMDB with Logistic Regression using ...

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config
from datasets.imdb import Imdb


from models.embeddings import Embedding


# Load dataset
print('Loading dataset...')
imdb = Imdb(config.DATASETS_FOLDER)
(train_x_texts, train_y), (test_x_texts, test_y) = imdb.get_texts_and_categories()


print('Loading Embeddings...')
NUM_WORDS = 5000
embeddings = Embedding(max_words = NUM_WORDS)


train_x_bof = embeddings.getTextAsBoF(train_x_texts)
print(train_x_bof.shape) 

test_x_bof = embeddings.getTextAsBoF(test_x_texts)
print(test_x_bof.shape)

# Train LR model
print('Training model...')
lm = LogisticRegression()
lm.fit(train_x_bof, train_y)

# Predict and score on test set
ps = lm.predict(test_x_bof)
acc = accuracy_score(test_y, ps)
print(f'Accuracy: {acc}')



