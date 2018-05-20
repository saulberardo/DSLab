# IMDB Dataset

## Attributes 
* Classes: 2
* Train size: 25k
* Test size: 25k


## Source
* Year: 2011
* Authors: Andrew L. Maas et al.
* Paper: Learning Word Vectors for Sentiment Analysis
* URL: https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf


## Benchmarks:
Accuracy | Paper | Model | URL |
--------:|:-----:|:-----:|-----|
88.89 | Add’l Unlabeled + BoW  | [Andrew L. Maas et al. 2011] | https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf
91.80 | CN+Char+CoVe | [McCann et al., 2017]  | https://arxiv.org/pdf/1708.00107.pdf
95.40 | ULMFit       | [Howard & Ruder, 2018] | https://arxiv.org/pdf/1801.06146.pdf

## Comments
* In [McCAnn et al, 2017] there are references to other paper with accuracies as high as 94.1.

* I don't know if ther results in the Benchmarks above are totaly comparable due to possible
differences in test setup.

* In Kaggle's IMDB toy competition LB, there are accuracies as high as 0.99259, but
possibly they were achieved with emsembles, not single models as in the Benchmarks
(URL: https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard).
