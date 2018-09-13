import pandas as pd
import os
import h5py
import re
from nltk.tokenize import RegexpTokenizer
from collections import Counter
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


def build_vocabs(all_words, max_words_num):
    assert isinstance(all_words, (str, list))
    max_words_num=max_words_num if max_words_num and max_words_num >= 5000 else 50000
    words_counter = Counter(all_words)
    word2id = {}
    id2word = {}
    word2id['PAD']=0
    word2id['EOS']=max_words_num+1
    word2id['SOS']=max_words_num+2
    word2id['UNK'] = max_words_num+3
    id2word[0], id2word[max_words_num+1], id2word[max_words_num+2], id2word[max_words_num+3]='PAD', 'EOS', 'SOS', 'UNK'
    i = 1
    words_mapping = dict(words_counter.most_common(max_words_num))
    for key in words_mapping.keys():
        id2word[i] = key
        word2id[key] = i
        i += 1
    return id2word, word2id