"""
TFRecordHelper: CreateTFRecordFile.py

Creates TFRecord file from .txt list for train and val sets
Requires args, default parsed in ArgsParser file

TFRecord code based on the following tutorial:
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    
And some inspiration taken from:
    https://github.com/NanqingD/DeepLabV3-Tensorflow

Muhammad Asad
"""


###############################################################################
#                 Helper functions for creating TFRecord Files
###############################################################################
#
# The following functions help in creating {train, val}.tfrecord files that contain
# binary data related to images, annotations, and their sizes
#
# --data-dir can be specified as input argument to the dataset creation file

from PIL import Image
import numpy as np
import tensorflow as tf
import os

from libs.ArgsParser import *

# create byte feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# create int64 features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# create TFRecord files
# takes input train.txt and val.txt files with the list of training and validation data
# helper functions in CreateTextFile can be used to create train.txt and test.txt
# files - they assume a folder structure (see CreateTextFile.py for more details)
def create_tfrecord_files():
    
    # get the input args | for --data-dir
    inArgs = parseArguments()
    
    # train and val file names
    inTxtFiles = ['train', 'val']
    
    # do the same operation for both txt files
    for inTxtFile in inTxtFiles:
        
        # create the path to store TFRecord files
        tfrecords_filename = os.path.join(inArgs.data_dir, '{}.tfrecords'.format(inTxtFile))
        print(tfrecords_filename)
        # initialize a TFRecordWriter for writing to file        
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        
        # open the corresponding text file (train/val)
        with open(os.path.join(inArgs.data_dir, '{}.txt'.format(inTxtFile)), 'r') as f:
            # read each line
            count = 0
            for line in f:
                line = line.strip()
                # split the paths for input and target images - split on " " space
                img_path, annotation_path = line.split()
                
                # read the images using PIL
                img = np.array(Image.open(img_path))
                annotation = np.array(Image.open(annotation_path))

                print("TFRecords %d \n %s file with data: \n %s" % (count, inTxtFile, line))
                count +=1
                
                # reading size of each image
                height = img.shape[0]
                width = img.shape[1]
                
                # getting the row bytes for each image
                img_raw = img.tostring()
                annotation_raw = annotation.tostring()
                
                # create a train.Example - that encapsulates all our data
                # shape, raw image and raw annotation data
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(img_raw),
                    'mask_raw': _bytes_feature(annotation_raw)}))
                
                # write each example
                writer.write(example.SerializeToString())
            
            # close the writer
            writer.close()    
            
# test TFRecords
# function used for verifying if the written data is correctly stored
# this function is optional to run, but will make sure you have no errors
# if there are error it will pick those
def check_tf_records_original_images():
     # get the args | for getting --data-dir path
    inArgs = parseArguments()
    
    #train and val file names
    inTxtFiles = ['train', 'val']
    
    # check for both train and val sets - same operations for both
    for inTxtFile in inTxtFiles:
        
        # creating list to store both images from TFRecord (reconstructed)
        # and images from disk (original)
        original_images = []
        reconstructed_images = []
        
        # read files from disk - use {train, val}.txt as input
        with open(os.path.join(inArgs.data_dir, '{}.txt'.format(inTxtFile)), 'r') as f:
            for line in f:
                line = line.strip()
                img_path, annotation_path = line.split()
                img = np.array(Image.open(img_path))
                annotation = np.array(Image.open(annotation_path))
                original_images.append((img, annotation))
        
        
        # path to tf record file
        tfrecords_filename = os.path.join(inArgs.data_dir + '{}.tfrecords'.format(inTxtFile))
        
        # iterator to read each sample
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
        
        # read images from TFRecords file (train, val).tfrecords
        for string_record in record_iterator:
            
            # get a single example
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            # extract relevant information for height, width, img and annotation
            height = int(example.features.feature['height']
                                         .int64_list
                                         .value[0])
            
            width = int(example.features.feature['width']
                                        .int64_list
                                        .value[0])
            
            img_string = (example.features.feature['image_raw']
                                          .bytes_list
                                          .value[0])
            
            annotation_string = (example.features.feature['mask_raw']
                                        .bytes_list
                                        .value[0])
            
            # convert string to uint8 image data
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
            
            # reshape both images to match the original size            
            reconstructed_img = img_1d.reshape((height, width, -1))
            # Annotations don't have depth (3rd dimension)
            reconstructed_annotation = annotation_1d.reshape((height, width))
            
            # append list with reconstructed images
            reconstructed_images.append((reconstructed_img, reconstructed_annotation))
            
            
        # compare the original with TFRecords
        
        # bit for check if all tests passed
        retVal = True
        for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
            img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)
            retVal &= np.allclose(*img_pair_to_compare)
            retVal &= np.allclose(*annotation_pair_to_compare)
            print(np.allclose(*img_pair_to_compare))
            print(np.allclose(*annotation_pair_to_compare))
            
        
        if(retVal):# all tests passed
            print(inTxtFile + ': All samples matched!!')
        else:# some tests failed
            print(inTxtFile + ': Some samples didnt match, check TFRecord creation')

# main function to debug the code independent of other components in the pipline
if __name__ == '__main__':
    
    # create TFRecords files
    create_tfrecord_files()
    
    # verify TFRecords files with original data
    check_tf_records_original_images()
    
    print('\nFinished converting the dataset to tfrecords files!')