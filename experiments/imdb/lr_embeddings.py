# -*- coding: utf-8 -*-
"""
Classify IMDB with Logistic Regression using using word embeddings (Bag of Features).

# Accuracy: 
    0.76628 (with 5k words)
    0.80664 (with 50k words)
    0.81732 (with 250k words)
    0.81828 (with 500k words)
    0.81892 (with entire vocab)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config
from datasets.imdb import Imdb



# Load dataset
print('Loading dataset...', flush=True)
imdb = Imdb(config.DATASETS_FOLDER)
(train_x_bof, train_y), (test_x_bof, test_y) = imdb.get_bof_fasttext_wiki_news_300d_1M()


# Train LR model
print('Training model...', flush=True)
lm = LogisticRegression()
lm.fit(train_x_bof, train_y)

# Predict and score on test set
ps = lm.predict(test_x_bof)
acc = accuracy_score(test_y, ps)
print(f'Accuracy: {acc}', flush=True)



