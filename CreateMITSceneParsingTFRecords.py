"""
FCNTFRecords: CreateMITSceneParsingTFRecords.py

uses TFRecordsHelper for creating TFRecord data files

Muhammad Asad
"""


from libs.ArgsParser import *
from libs.datasetlibs.CreateTextFileList import *
from libs.datasetlibs.CreateTFRecordFile import *


# main function to debug the code independent of other components in the pipline
if __name__ == '__main__':
    create_text_file_with_filenames()
    print('\nFinished converting the dataset to text files!')
    # create TFRecords files
    create_tfrecord_files()
    print('\nFinished converting the dataset to tfrecords files!')
    
    