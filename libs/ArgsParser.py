import argparse

# function for argument parsing - setting some defaults for when it is run from
# spyder
def parseArguments():
    DATA_DIRECTORY = "data"
    INPUT_SIZE = 513
    BATCH_SIZE = 5
    #SPLIT_NAME = "train"
    
    parser = argparse.ArgumentParser(description="FCN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
    #parser.add_argument("--split-name", type=str, default=SPLIT_NAME,
    #                help="Split name.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="height and width of images.")
    
    args = parser.parse_args()
    
    return args