import h5py
import pandas as pd
import os
import random as rn
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.create_vocab import preprocess

def create_datesets():
    """
    Args:
        the high level function to create a h5py file,
        mode can be either train, val or test, for the 
        train mode, one can set the datasets to be a content
        and each word will be mapping to a index, if the word
        not in the vocabulary, then set skip this or set to
        Unkow regard to the model's setting. 
    """
    pass

def word2id(vocab, sentence, skip=True):
    ids=[]
    for word in sentence:
        try:
            ids.append(vocab[word])
        except KeyError:
            if skip:
                pass


            
