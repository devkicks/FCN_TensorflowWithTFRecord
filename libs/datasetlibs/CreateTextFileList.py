"""
TFRecordHelper: CreateTextFileList.py

TFRecord code based on the following tutorial:
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    
And some inspiration taken from:
    https://github.com/NanqingD/DeepLabV3-Tensorflow

Muhammad Asad
"""

###############################################################################
#                       Helper functions for creating text files
###############################################################################
#
# The following functions help in creating {train, val}.txt files that contain
# image paths in the following format:
#
#            /data/{train, val}/images   /data/{train, val}/annotations
#
#
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
import sys

# adding dir for command line execution
sys.path.insert(0, os.path.join('..','..'))

from libs.ArgsParser import *
   
# function gets the list of filesnames and stores them to train.txt and test.txt
def create_text_file_with_filenames():
    
    inArgs = parseArguments()
    
    all_files_list = get_image_lists(inArgs.data_dir)
    
    trainFilenames = all_files_list['train']
    valFilenames = all_files_list['val']
    
    # now that we have two lists - save them on disk, so that they can be used
    # by the tfrecord builder later
    train_fileName = os.path.join(inArgs.data_dir, 'train.txt')
    #print(train_fileName)
    with open(train_fileName, 'w') as f:
        for item in trainFilenames:
            f.write("%s\n" % item)    
    
    val_fileName = os.path.join(inArgs.data_dir, 'val.txt')
    with open(val_fileName, 'w') as f:
        for item in valFilenames:
            f.write("%s\n" % item)    
    
    # All done -
#    return trainFilenames, valFilenames

# create a list of filenames with complete path as follows
# colorimage  gtannotationimage
def get_image_lists(image_dir):
    if not os.path.isdir(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train', 'val']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = list()
        count = 0
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        #print(file_glob)
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split(os.path.sep)[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
				
                if os.path.exists(annotation_file):
                    record = f + " " + annotation_file 
                    print("Appending %d/%d \n %s file with: \n %s" % (count, len(file_list), directory, record))
                    #{'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
                count +=1

        #random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    # we must have read everything by now, check if it all looks alright?
    return image_list

# main function to debug the code independent of other components in the pipline
if __name__ == '__main__':
  create_text_file_with_filenames()
  print('\nFinished converting the dataset to text files!')
