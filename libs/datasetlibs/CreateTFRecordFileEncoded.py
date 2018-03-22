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
import matplotlib.pyplot as plt

from libs.ArgsParser import *

# create byte feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# create int64 features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# build example - maybe modify when image formats are diffferent
def _convert_to_example(image, label, width, height):
    image_format = "JPEG"
    label_format = "PNG"
    example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
                    'image/encoded': _bytes_feature(tf.compat.as_bytes(image)),
                    'mask/format': _bytes_feature(tf.compat.as_bytes(label_format)),
                    'mask/encoded': _bytes_feature(tf.compat.as_bytes(label))}))
    return example

# create TFRecord files
# takes input train.txt and val.txt files with the list of training and validation data
# helper functions in CreateTextFile can be used to create train.txt and test.txt
# files - they assume a folder structure (see CreateTextFile.py for more details)
def create_tfrecord_files():
    
    # get the input args | for --data-dir
    inArgs = parseArguments()
    
    # train and val file names
    inTxtFiles = ['train', 'val']
    annotation_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_annotation = tf.image.encode_png(tf.expand_dims(annotation_placeholder, 2))
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(image_placeholder, quality=80)
    with tf.Session() as sess:
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
                    
                    
                    # read image with tf
#                    with tf.gfile.FastGFile(img_path, 'rb') as ff:
#                        img_raw = ff.read()
                    
                    img = np.array(Image.open(img_path))
                    
                    # our model expects 3 channels (i.e. color images)
                    # so here we check if grayscale image - then duplicate to three channels
                    if (len(img.shape) != 3):
                        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                    
                    # read the images using PIL
                    annotation = np.array(Image.open(annotation_path))
    
                    print("TFRecords %d \n %s file with data: \n %s" % (count, inTxtFile, line))
                    count +=1
                    
                    # reading size of each image
                    height = annotation.shape[0]
                    width = annotation.shape[1]
                    
                    # getting the raw bytes for each image
                    img_raw, annotation_raw = sess.run([encoded_image, encoded_annotation], feed_dict={image_placeholder: img, annotation_placeholder : annotation})
                    
                    # create a train.Example - that encapsulates all our data
                    # shape, raw image and raw annotation data
                    example = _convert_to_example(img_raw,  annotation_raw, width, height)
                    
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
        tfrecords_filename = os.path.join(inArgs.data_dir, '{}.tfrecords'.format(inTxtFile))
        
        # iterator to read each sample
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
        
        # decoder for image
        encoded_jpeg= tf.placeholder(dtype=tf.string)
        decoded_jpeg = tf.image.decode_jpeg(encoded_jpeg, channels=3)
        
        # decoder for annotations
        encoded_png = tf.placeholder(dtype=tf.string)
        decoded_png = tf.image.decode_png(encoded_png, channels=1)
        
        with tf.Session() as sess:
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
                
                image_format = tf.compat.as_str(example.features.feature['image/format']
                                            .bytes_list
                                            .value[0])
                
                annotation_format = tf.compat.as_str(example.features.feature['mask/format']
                                            .bytes_list
                                            .value[0])
                
                img_raw = (example.features.feature['image/encoded']
                                              .bytes_list
                                              .value[0])
                
                annotation_raw = (example.features.feature['mask/encoded']
                                            .bytes_list
                                            .value[0])
                
                
                # decode images and annotations
                
                reconstructed_annotation = sess.run(decoded_png, feed_dict={encoded_png: annotation_raw})
                reconstructed_img = sess.run(decoded_jpeg, feed_dict={encoded_jpeg: img_raw})
                
                # convert string to uint8 image data
    #            img_1d = np.fromstring(img_string, dtype=np.uint8)
    #            annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
                
                # reshape both images to match the original size            
    #            reconstructed_img = img_1d.reshape((height, width, -1))
                # Annotations don't have depth (3rd dimension)
    #            reconstructed_annotation = annotation_1d.reshape((height, width))
                
                # append list with reconstructed images
                reconstructed_images.append((reconstructed_img, np.squeeze(reconstructed_annotation)))
            
            
        # compare the original with TFRecords
        
        # bit for check if all tests passed
        retVal = True
        for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
            img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)
            
            # debugging - show the actual images
            plt.subplot(1, 3, 1)
            plt.imshow(img_pair_to_compare[0])
            plt.subplot(1, 3, 2)
            plt.imshow(img_pair_to_compare[1])
            #plt.subplot(1, 3, 3)
            #plt.imshow(img_pair_to_compare[1]-img_pair_to_compare[0])
            plt.savefig('savedFig.png')
            plt.show()
#            plt.show()
            
            result = Image.fromarray(img_pair_to_compare[0])
            result.save('out0.png')
            result = Image.fromarray(img_pair_to_compare[1])
            result.save('out1.png')
            
            print("Sum is: " + str(np.sum(img_pair_to_compare[0] - img_pair_to_compare[1])))
            print("Max image1: " + str(np.max(img_pair_to_compare[0])))
            print("Max image2: " + str(np.max(img_pair_to_compare[1])))
            retVal &= np.allclose(*img_pair_to_compare)
            retVal &= np.allclose(*annotation_pair_to_compare)
            print("Image: " + str(np.allclose(*img_pair_to_compare)))
            print("Annotation: " + str(np.allclose(*annotation_pair_to_compare)))
            
            
        
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