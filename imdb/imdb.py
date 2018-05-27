# -*- coding: utf-8 -*-
"""

"""

import math
import os
from tqdm import tqdm
import requests
import tarfile


def download_file(url, out_file, chunk_size=1024):
    """
    Download file from url and save in the destinatin path.
    """
    # Get file headers
    response = requests.get(url, stream=True)
    
    # Get file length and determine the number of chunks that will be downloaded
    file_length = int(response.headers.get('content-length'))
    number_of_chunks = math.ceil(file_length//chunk_size) 
    
    # Open file
    with open(out_file, "wb") as f:
        
        #  Write each chunk at a time        
        for data in tqdm(response.iter_content(chunk_size), total=number_of_chunks, unit='KB', unit_scale=True):    
            f.write(data)

    
def un_gzip_tar_file(in_file, destination_folder=None):
    """ Extract content from in_file and save it in destination_folder (or . if None is passed)"""
    destination_folder = os.path.dirname(in_file) if destination_folder is None else destination_folder    
    tfile = tarfile.open(in_file, 'r:gz')
    tfile.extractall(destination_folder)

if __name__=='__main__':
 
    DESTINATION_FOLDER = '..'
    DESTINATION_FILE = 'aclImdb_v1.tar.gz'
    IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    
    # Download file if it doesn't already exists
    download_output = os.path.join(DESTINATION_FOLDER, DESTINATION_FILE)    
    if not os.path.exists(download_output):        
        print(f'Downloading file to: {download_output}')
        download_file(IMDB_URL, download_output)
    
    # Extract files from tar.gz
    aclImdb_folder = os.path.join(DESTINATION_FOLDER, 'aclImdb')
    if not os.path.exists(aclImdb_folder):
        print(f'Extracting files to: {aclImdb_folder}')
        un_gzip_tar_file(download_output, DESTINATION_FOLDER)
    
    
    
    
    
    
    