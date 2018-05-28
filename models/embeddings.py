# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:02:05 2018

@author: camposb
"""

import os
import config
import numpy as np
from util import download_file, un_zip_file


class Embedding:
    
    VOCAB_SIZE = 999995
    
    def __init__(self, embedding='fasttext_wiki_news_300d_1M', max_words=None):
        """ Instantiate a word embedding.
        
        Parameters
        ----------
        max_words : int
            Maximum number of features that will be loaded.
            
        embedding : str
            A string specifying which word embeding to load (possible values:
            `fasttext_wiki_news_300d_1M`).
        """
        if not max_words is None:            
            assert(max_words <= self.VOCAB_SIZE)
        else:
            max_words = self.VOCAB_SIZE
        self._max_words = max_words
        
        # Load corret emebddings acordingly to the parameters
        if embedding == 'fasttext_wiki_news_300d_1M':
            self.emebed_file = self._download_fasttext_wiki_news_300d_1M()
        else:
            raise Exception(f'Error: embedding {embedding} not found.')
        
    
    
    def _download_fasttext_wiki_news_300d_1M(self):
        """ Download and extract contents from FastText embeddings repository."""
        
        _EMBEDD_URL = 'http://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip'
    
        # wiki-news-300d-1M.vec folder
        wiki_news_300d_1M_vec_file = os.path.join(config.EMBEDDINGS_FOLDER, 'wiki-news-300d-1M.vec')
        
        # File where the downloaded aclImdb_v1.tar.gz will be saved
        download_output = os.path.join(config.EMBEDDINGS_FOLDER, 'wiki-news-300d-1M.vec.zip')   
        
        # Download file if the dataset isn't avaiblable in the dataset folder
        if not os.path.exists(wiki_news_300d_1M_vec_file) and not os.path.exists(download_output):        
            print(f'Downloading file to: {download_output}')        
            download_file(_EMBEDD_URL, download_output)
            
        # Extract files from .zip  
        if not os.path.exists(wiki_news_300d_1M_vec_file):
            print(f'Extracting files to: {wiki_news_300d_1M_vec_file}')
            un_zip_file(download_output, config.EMBEDDINGS_FOLDER)        
                    
        # Set the embedding dimension
        self.embdding_dim = 300
            
        # return embedding file
        return wiki_news_300d_1M_vec_file
    
        
        
    def getEmbedDict(self):
        """ 
        Return dict mapping word to embeddings for the first num_word most frequent words.
        """
        # Return the dict if it has already been  loaded
        try:
            return self._m        
        except:
            pass
        
        # Dictionary to store word : emebdding pairs
        self._m = {}
        
        with open(self.emebed_file, encoding='utf8') as f:    
            
            # For i rangin from 0 to the num_words that will be loaded
            for i in range(0, self._max_words):
                
                # Extract word : embedding values from line and store in the dict              
                line = next(f)
                values = line[0:-1].split(' ')       
                
                self._m[values[0]] = np.array(values[1::], dtype=np.float32)
            
        return self._m

    
    def getTextAsLoE(self, texts):
        """ Return the text string as a LoE (List of Embeddings). 
        
        Parameters
        ----------
        texts : str or list/tuple of str.
            Text or list/tuple of texts.
            
        Returns
        -------
        List of LoE : list
            List of texts converted to LoE.
        
        TODO: For unkown words, we're using a vector full with zeros. Perhaps
        we should change this.
        """
        m = self.getEmbedDict()
        
        # List of LoEs (List of Embeddings) for each text
        loes = []
        
        # Wrap text in a list, in the case it is not in a list or tuple lready
        if not type(texts) in [list, tuple]: texts = [texts]
        
        # For each text
        for text in texts:      
            
            # LoE of a single text
            loe = []    
            
            # For each word
            for word in text.split():        
                try:
                    # Add word to corpora_embeds list
                    loe.append(m[word])
                except:
                    loe.append(np.zeros(self.embdding_dim, dtype=np.float32))
                    
            # Add text embeds to corpora_embeds list
            loes.append(np.array(loe))
            
        return np.array(loes)
    
    
    def getTextAsBoF(self, texts):
        """ Return the text string as a BoF (Bag of Features) -- i.e. mean of 
        embeddings of words contained in the text.
        
        Return
        ------
        Texts as BoF : np.ndarray
                
        TODO: For unkown words, we're using a vector full with zeros. Perhaps
        we should change this.
        """        
        # Wrap text in a list, in the case it is not in a list or tuple lready
        if not type(texts) in [list, tuple]: texts = [texts]
        
        bofs = []
        
        # For each text
        for text in texts:                
            
            # Get List of Embeddings
            loe = self.getTextAsLoE(text)
            
            # Compute mean row-wise of each features (Bag of Features)
            bof = loe[0].mean(0)
            
            # Append to lsit
            bofs.append(bof)
            
        return np.array(bofs)
        
    
    
e = Embedding()
m = e.getEmbedDict()
l = e.getTextAsLoE('this is a test')