import pandas as pd
import os
import h5py
import re
"""
create a simple vocab files, it will be stored in the ../data folder with a
json file. This script will be the baseline file to create the Voacb. 
"""

def create_vocab(datapath = None):
    """
    Args:
        datapath, will be the processed data, like we have use the re 
        module to sub some unwanted punctuius
    """

