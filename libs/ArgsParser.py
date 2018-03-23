import argparse
import os
# function for argument parsing - setting some defaults for when it is run from
# spyder
def parseArguments():
    #DATA_DIRECTORY = "Data_zoo"
#    DATA_DIRECTORY = os.path.join("Data_zoo", "MIT_SceneParsing", "ADEChallengeDataSmall")
#    DATA_DIRECTORY = os.path.join("Data_zoo", "MIT_SceneParsing", "ADEChallengeData2016")
    DATA_DIRECTORY = os.path.join("Data_zoo", "MIT_SceneParsing", "ADEChallengeData2016")
    INPUT_SIZE = 224
    BATCH_SIZE = 2
    MODEL_DIR = "Model_zoo"
    LOG_DIR = "logs"
    LEARNING_RATE = 1e-4
    MODE = "train"
    DEBUG = False
    DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'    
    
    
    parser = argparse.ArgumentParser(description="FCN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the dataset.")
    #parser.add_argument("--split-name", type=str, default=SPLIT_NAME,
    #                help="Split name.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="height and width of images.")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                    help="Path to VGG model mat")
    
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                    help="Path to the logs directory")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                    help="Learning rate for training.")
    parser.add_argument("--mode", type=str, default=MODE,
                    help="train or val mode.")
    parser.add_argument("--debug", type=bool, default=DEBUG,
                    help="Debug True or False")		
    parser.add_argument("--data-url", type=str, default=DATA_URL,
                    help="URL to the dataset used")			
    
    args = parser.parse_args()
    
    return args


# main function to debug the code independent of other components in the pipline
if __name__ == '__main__':
  args = parseArguments()
  
  print("data_dir: " + args.data_dir)
  print("batch_size: " + str(args.batch_size))
  print("input_size: " + str(args.input_size))
  print("model_dir: " + args.model_dir)
  print("log_dir: " + args.log_dir)
  print("learning_rate: " + str(args.learning_rate))
  print("mode: " + args.mode)
  print("debug: " + str(args.debug))
  
  print('\nFinished testing args parser!')