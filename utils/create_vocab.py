import pandas as pd
import os
import h5py
import re
from nltk.tokenize import RegexpTokenizer
"""
create a simple vocab files, it will be stored in the ../data folder with a
json file. This script will be the baseline file to create the Voacb. 
"""

def find_all_vocabs(file_path):
    """
    Args:
        file_path: a csv file in  the ../data folder
    """
    content_words=[]
    title_words=[]
    data = pd.read_csv(file_path, encoding='utf-8')
    for item in list(data.content):
        tokens=list(preprocess(item))
        content_words.extend(tokens)

    for item in list(data.title):
        tokens=list(preprocess(item))
        title_words.extend(tokens)
    all_words = content_words+title_words
    del content_words, title_words
    return all_words



def preprocess(sentences):
    sentences = sentences.lower()
    tokenizer = RegexpTokenizer(r'w\+')
    tokens = tokenizer.tokenize(sentences)
    filtered_words = list(tokens)
    return filtered_words

