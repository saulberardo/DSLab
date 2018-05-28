# -*- coding: utf-8 -*-
"""
Class to download IMDB and load train and test sets.

"""

import os
import glob
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from util import download_file, un_gzip_tar_file


class Imdb:
    
    # IMDB file name and URL    
    _IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"    
    
    
    def __init__(self, datasets_folder):
        """ Download IMDB dataset and load train and test sets"""
        
        # Folder containing the original dataset
        aclImdb_folder = os.path.join(datasets_folder, 'aclImdb', 'original')
        
        # File where the downloaded aclImdb_v1.tar.gz will be saved
        download_output = os.path.join(aclImdb_folder, 'aclImdb_v1.tar.gz')        
                
        # Download file if the dataset isn't avaiblable in the dataset folder
        if not os.path.exists(aclImdb_folder) and not os.path.exists(download_output):        
            print(f'Downloading file to: {download_output}')
            download_file(self._IMDB_URL, download_output)
        
        # Extract files from tar.gz  
        if not os.path.exists(os.path.join(aclImdb_folder, 'aclImdb')):
            print(f'Extracting files to: {aclImdb_folder}')
            un_gzip_tar_file(download_output, aclImdb_folder)
            
        # Assing datasets_folder to instance
        self.aclImdb_folder = aclImdb_folder
         

    def _get_texts_and_categories(self, train_or_test):  
        """ Return texts and categories from the specified dataset (train or test)"""
        texts = []
        categories = []
        for pos_or_neg in ['pos', 'neg']:
            files = glob.glob(os.path.join(self.aclImdb_folder, 'aclImdb', train_or_test, pos_or_neg, '*.txt'))
            for pos_file in files:
                id, score = re.findall('(\d+)_(\d+).txt', pos_file)[-1]
                with open(pos_file, encoding='utf8') as f:
                    text = f.read()
                
                category = 1 if pos_or_neg == 'pos' else 0
                categories.append(category)
                texts.append(text)
        return (texts, categories)
    
    
    def get_texts_and_categories(self):
        """ Return train and test sets.
        
        Returns
        -------
        (train_texts, train_categories), (test_texts, test_categories) : tuple of tuples
        """
        return self._get_texts_and_categories('train'),  self._get_texts_and_categories('test')
            
    
    def get_bow_and_categories(self, max_features, **kwargs):
        """ Convert texts to BoW and return train and test sets.
        
        Parameters
        ----------
        All parameters are forwarded to sklearn CountVectorizer.
            
        Returns
        -------
        (train_x_bow, train_categories), (test_x_bow, test_categories) : tuple of tuples
            
        """
        # If max_features is None, use the entire vocab used by IMDB
        max_features = 74849 if max_features is None else max_features
        
        # Pickled numpy dataset file
        numpy_dataset_file = os.path.join(self.aclImdb_folder, os.path.pardir, f'imdb_bow_{max_features}.npy')
        
        # If the pickled file doesn't exist
        if not os.path.exists(numpy_dataset_file):
            
            print(f'Converting text to BoW and saving to: {numpy_dataset_file}')
            
            # Load texts from files
            (train_texts, train_categories), (test_texts, test_categories) = self.get_texts_and_categories()
            
            # Create vectorizer to convert texts to BoW
            cv = CountVectorizer(max_features=max_features, **kwargs)
            cv.fit(train_texts)
            
            # Convert texts to BoW
            train_x_bow = cv.transform(train_texts)
            test_x_bow = cv.transform(test_texts)
            
            # Put together all dataset variables
            data = { 'train':(train_x_bow, train_categories), 'test':(test_x_bow, test_categories)}
            
            # Save dataset to pickle file
            np.save(numpy_dataset_file, data)
        else:
            # Load pickled data
            data = np.load(numpy_dataset_file).item()
                        
        return data['train'], data['test']
        
             

    