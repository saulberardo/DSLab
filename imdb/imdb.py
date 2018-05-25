# -*- coding: utf-8 -*-
"""

"""

import math
import os
from tqdm import tqdm
import requests


def download_file(url, destination_folder='.', chunk_size=1024):
    """
    Download file from url and save in the destinatin path.
    """
    # Get file headers
    response = requests.get(url, stream=True)
    
    # Get file length and the number of chunks that will be downlaoded
    file_length = int(response.headers.get('content-length'))
    number_of_chunks = math.ceil(file_length//chunk_size) 
    
    # Get the file name from the URL
    file_name = url.split('/')[-1]
    
    # Open file
    with open(os.path.join(destination_folder, file_name), "wb") as f:
        
        #  Write each chunk at a time        
        for data in tqdm(response.iter_content(chunk_size), total=number_of_chunks, unit='KB', unit_scale=True):    
            f.write(data)


if __name__=='__main__':
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    download_file(url, '..')