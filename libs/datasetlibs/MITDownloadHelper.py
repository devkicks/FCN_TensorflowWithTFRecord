
"""


Muhammad Asad
"""

###############################################################################
#                       Helper functions for loading MIT dataset
###############################################################################
#
# The following function help in  downloading and naming data folders in the
# following directory structure
# To use these functions out of the box, you will need to create the following
#  directory structure
#
#
""" 
     ├── --data-dir
     |   ├── images
     |   |   ├── train
     |   |   ├── val
     |   ├── annotations
     |   |   ├── train
     |   |   ├── val
"""
# --data-dir can be specified as input argument to the dataset creation file
import os
import glob
import numpy as np

from six.moves import urllib
import tarfile
import zipfile
import scipy.io

import sys
# adding dir for command line execution
sys.path.insert(0, os.path.join('..','..'))


from libs.ArgsParser import *

# function downloads (if required) and renames the train and val folders
def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=True):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (url_name, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, dir_path, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', url_name, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)
# function renames train / val folders if they are names training or validation
# works for MIT dataset
def maybe_rename_folders(data_dir, url_name):    
    if not os.path.exists(data_dir):
        print('Error: data_dir incorrect/ does not exist')
        return
    else:
        filename = url_name.split('/')[-1]
        foldername = filename.split('.')[0]
        folderpath = os.path.join(data_dir, foldername)
        
        imagefolder = os.path.join(folderpath, 'images')
        annotationfolder = os.path.join(folderpath, 'annotations')
        folders = [imagefolder, annotationfolder]
        # renaming folders if required
        for i in folders:
            folder_list = []
            folder_list.extend(glob.glob(i))
            
            for j in folder_list:
                print(j)
            

        
# main function to debug the code independent of other components in the pipline
if __name__ == '__main__':
    inArgs = parseArguments()
    print("Data url: " + inArgs.data_url)
    print("Data dir: " + inArgs.data_dir)
    maybe_download_and_extract(inArgs.data_dir, inArgs.data_url)
    maybe_rename_folders(inArgs.data_dir, inArgs.data_url)
    print('\nFinished downloading the dataset!')
