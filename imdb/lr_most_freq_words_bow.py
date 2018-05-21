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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# only use top 1000 words ( I don't know why, but it uses always 10 less words than specified here)
NUM_WORDS = 1000

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
train_x_bow = cv.transform(np.array(train_x_str))
test_x_bow = cv.transform(np.array(test_x_str))

# Train LR model
lm = LogisticRegression()
lm.fit(train_x_bow, train_y)

# Predict and score on test set
ps = lm.predict(test_x_bow)
acc = accuracy_score(test_y, ps)
print(acc)



def indices_to_words(indices_text):
    """
    Return text string corresponding the word indices sequence passed as paramenter.
    
    Source: https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    """
    # Get word_to_id map
    word_to_id = keras.datasets.imdb.get_word_index()
    
    # Altough the default INDEX_FROM used by Keras is 3, the min index returned by
    # word_to_id is 1, as we can verify reunning:
    # np.array(list(keras.datasets.imdb.get_word_index().values())).min()
    
    # Shift forward the ID of every word in the map (add INDEX_FROM to them)
    word_to_id = { k : (v + INDEX_FROM) for k, v in word_to_id.items()}
    
    # Add special tags to indexes (these are the default values in load_data(...))
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    
    # Create id_to_word map
    id_to_word = {value:key for key,value in word_to_id.items()}
    
    # Print the text of one training sample
    return ' '.join(id_to_word[id] for id in indices_text)
    
print(indices_to_words(train_x[0]))


