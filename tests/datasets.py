# -*- coding: utf-8 -*-
"""
Test cases for datasets.
"""

import unittest
import config

from datasets.imdb import Imdb

class TestIMDB(unittest.TestCase):
    
    def test_get_texts_and_categories(self):                
        imdb = Imdb(config.DATASETS_FOLDER)        
        (train_texts, train_categories), (_, _) = imdb.get_texts_and_categories()
        self.assertEqual(len(train_texts), 25000)
        self.assertEqual(len(train_categories), 25000)
    
    def test_get_bow_and_categories(self):
        imdb = Imdb(config.DATASETS_FOLDER)        
        num_features = 5000
        (train_x_bow, train_categories), (_, _) = imdb.get_bow_and_categories(max_features=num_features)
        self.assertEqual(train_x_bow.shape, (25000, num_features))
        self.assertEqual(len(train_categories), 25000)
    
if __name__ == '__main__':
    unittest.main()
        