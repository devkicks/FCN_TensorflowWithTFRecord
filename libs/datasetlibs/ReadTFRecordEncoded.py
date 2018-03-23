"""
TFRecordHelper: ReadTFRecord.py

TFRecord code based on the following tutorial:
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    
And some inspiration taken from:
    https://github.com/NanqingD/DeepLabV3-Tensorflow

Reader help taken from: 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

Muhammad Asad
"""

###############################################################################
#                 Helper functions for reading TFRecord Files 
#                 (encoded version with image compression)
###############################################################################
#
# The following functions helps in reading {train, val}.tfrecord files that contain
# binary data related to images, annotations, and their sizes
#
# it also provides a bare bones example for debugging and getting idea for how to use it
#
#
# map functions are also provided that can be added to pipline for preprocessing images
# e.g. resize, data augmentation, normalization etc.
#
# required parameters (all set to default in ArgsParser code)
# --data-dir (args.data_dir) can be specified as input argument to the dataset creation file
# --input-size (args.input_size) specifies size of input images (both width and height, assuming square)
# --batch-size (args.batch_size) specifies size of batch for training

import matplotlib.pyplot as plt
import tensorflow as tf
import os
from libs.ArgsParser import *

# decode tfrecord example
# helper function used to decode tfrecord example with the provided format
def decode(serialized_example):
    # parse data for individual containers    
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image/format': tf.FixedLenFeature([], tf.string),
                    'mask/format': tf.FixedLenFeature([], tf.string),
                    'image/encoded': tf.FixedLenFeature([], tf.string),
                    'mask/encoded': tf.FixedLenFeature([], tf.string)                    
                    })
    # Convert from a scalar string tensor to a uint8 tensor with shape (1D)
#    image = tf.decode_raw(features['image/encoded'], tf.uint8)
#    label = tf.decode_raw(features['mask/encoded'], tf.uint8)
    image_raw = features['image/encoded']
    label_raw = features['mask/encoded']
    image = tf.image.decode_jpeg(image_raw, channels=0)
    label = tf.image.decode_png(label_raw, channels=1)
    
    # OTHER information - not required at the moment
    # extract the height and width - this is useful for reshaping the image data
#    height = tf.cast(features['height'], tf.int32)
#    width = tf.cast(features['width'], tf.int32)
    
    # create shape tensors for image and annotation
#    image_shape = tf.stack([height, width, 3])
#    label_shape = tf.stack([height, width, 1])
    
    # reshape the image and annotation
#    image = tf.reshape(image, image_shape)
#    label = tf.reshape(label, label_shape)
    
    # NOT USED
    ##image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    ##label_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
    return image, label

# augment function - 
# mapping function for data augmentation or any preprocessing
def augment(image, label):
     
    # get the input arguments
    inArgs = parseArguments()
    
    # get the input shape for each image - modify this in ArgsParser or through
    # --input-shape option in command line
    new_height = inArgs.input_size
    new_width = inArgs.input_size
      
    # OPTIONAL: Could reshape into a HxWxC image and apply distortions
    # here.  Since we are not applying any distortions in this code
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                             target_height=new_height,
                                             target_width=new_height)
    label = tf.image.resize_image_with_crop_or_pad(image=label,
                                             target_height=new_height,
                                             target_width=new_width)
    return image, label


# normalize function 
# maps the images to normalized - not used
def normalize(image, label):
  image = tf.cast(image, tf.float32) #* (1. / 255) - 0.5
  label = tf.cast(label, tf.int32) #* (1. / 255) - 0.5
  return image, label

# input function 
# function for creating a placeholder with tfrecords data files
def inputs(mode, batch_size, num_epochs=None):
  """Reads input data num_epochs times.
  Args:
    mode: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  
  # parse the input arguments - used for --data-dir and --batch-size
  inArgs = parseArguments()

  # based on the input to train (True, False) select (train, val) tfrecords    
  inTxtFiles = ['train', 'val']
  inTxtFile = []
  if mode is not None: # if training set selected
      inTxtFile = inTxtFiles[0]
  else:
      inTxtFile = inTxtFiles[1]
    # create path to tfrecords file
  filename = os.path.join(inArgs.data_dir, '{}.tfrecords'.format(inTxtFile))

  # open data pipeline inside input scope
  with tf.name_scope('input'):
      
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode)
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)

    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    # repeat data infinitely (-1) - can be set to num_epochs
    dataset = dataset.repeat(-1)
    
    # read data in provided batch_size
    dataset = dataset.batch(batch_size)

    # return an iterator output (image, labels) to the main script - can be used to extract data
    # from tfrecords with the above defined data pipeline
    iterator = dataset.make_one_shot_iterator()
    
  return iterator.get_next()

# main function to debug the code independent of other components in the pipeline
if __name__ == '__main__':
    
    # get the input arguments
    inArgs = parseArguments()
    
    # make placeholder for data pipline
    image, label = inputs('train', inArgs.batch_size)
    
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session()  as sess:    
        sess.run(init_op)
        
        # read data in batches
        for i in range(30):
            # run graph
            img, anno = sess.run([image, label])
            
            # show size of the batch
            print(img[:, :, :, :].shape)
            
            print('current batch')
            
            # display images - WARNING NOT GOOD FOR LARGE DATA
            for j in range(img.shape[0]):
                plt.imshow(img[j, :, :, :])
                plt.show()
        
                plt.imshow(anno[j, :, :, 0])
                plt.show()