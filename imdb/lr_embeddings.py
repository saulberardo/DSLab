# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:31:49 2018

@author: camposb
"""


embeddings_file = 'D:\\dados\\embeddings\\en\\fasttext\\wiki-news-300d-1M.vec'


import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


embeddings = load_vectors(embeddings_file)