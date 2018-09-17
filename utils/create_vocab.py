import pandas as pd
import os
import h5py
import re
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pickle
"""
for small data, we use pickle to store them as pkl file, for complex and
huge file, we will sotre them as hdf5 file.
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
    filtered_words = [w for w in tokens]
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



if __name__ == '__main__':
    data_dir = '../data'
    csv_file = os.path.join(data_dir, 'test_data.csv')
    all_words = find_all_vocabs(csv_file)
    id2word, word2id = build_vocabs(all_words, 50000)
    Vocab = {'word2id':word2id, 'id2word':id2word}
    with open(os.path.join(data_dir, 'Vocab.pkl'), 'wb') as f:
        pickle.dump(Vocab, f)
    print(f"the data has been saved to{data_dir}")
