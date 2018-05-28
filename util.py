# -*- coding: utf-8 -*-
"""
Usefull methods to use across the project.
"""

import os
import requests
import tarfile
import zipfile
import math
import tqdm

def download_file(url, out_file, chunk_size=1024):
    """
    Download file from url and save in the destinatin path.
    """
    # Get file headers
    response = requests.get(url, stream=True)
    
    # Get file length and determine the number of chunks that will be downloaded
    file_length = int(response.headers.get('content-length'))
    number_of_chunks = math.ceil(file_length//chunk_size) 
    
    # Create output dir, if it doesn't exist
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        print(f'creating {out_dir}')
        v=os.makedirs(out_dir)
        print(v)
    
    # Open file
    with open(out_file, "wb") as f:
        
        #  Write each chunk at a time        
        for data in tqdm.tqdm(response.iter_content(chunk_size), total=number_of_chunks, unit='KB', unit_scale=True):    
            f.write(data)

    
def un_gzip_tar_file(in_file, destination_folder=None):
    """ Extract content from in_file and save it in destination_folder (or . if None is passed)"""
    destination_folder = os.path.dirname(in_file) if destination_folder is None else destination_folder    
    tfile = tarfile.open(in_file, 'r:gz')
    tfile.extractall(destination_folder)
    
def un_zip_file(in_file, destination_folder=None):
    """ Extract content from in_file and save it in destination folder (or . if None is passed)"""    
    destination_folder = os.path.dirname(in_file) if destination_folder is None else destination_folder    
    zip_ref = zipfile.ZipFile(in_file, 'r')
    zip_ref.extractall(destination_folder)
    zip_ref.close()