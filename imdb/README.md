# IMDB Dataset

## Attributes 
Classes: 2
Train size: 25k
Test size: 25k


## Source
Year: 2011
Authors: Andrew L. Maas et al.
Paper: Learning Word Vectors for Sentiment Analysis
URL: https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf


## Benchmarks:
88.89 - [Andrew L. Maas et al. 2011] - -            https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf
91.80 - [McCann et al., 2017] - BCN+Char+CoVe -     https://arxiv.org/pdf/1708.00107.pdf
95.40 - [Howard & Ruder, 2018] - ULMFit -           https://arxiv.org/pdf/1801.06146.pdf

# Comments
In [McCAnn et al, 2017] there are references to other paper with accuracies 
as highas 94.1.

I don't know if ther results in the Benchmarks are totaly comparable due to possible
differences in the test setup.

In Kaggle's IMDB toy competition LB, there are accuracies as high as 0.99259, but
possible these use emsembles or other stuff:
https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard