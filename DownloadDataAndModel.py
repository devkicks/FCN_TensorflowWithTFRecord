from __future__ import print_function
import numpy as np
import TensorflowUtils as utils

from libs.ArgsParser import *

inArgs = parseArguments()

def main(argv=None):
  
#    model_data = utils.get_model_data(inArgs.model_dir, MODEL_URL)
    utils.maybe_download_and_extract(inArgs.model_dir, inArgs.model_url, is_zipfile=True)
    utils.maybe_download_and_extract(inArgs.data_dir, inArgs.data_url, is_zipfile=True)
    

if __name__ == "__main__":
    main()
