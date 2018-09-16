import h5py
import pandas as pd
import os
import random as rn
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.create_vocab import preprocess
# notice here to add a config file, we need also to use the config
# to assure that we may need some oov in the model
import utils.config as config

def create_datesets(data):
    """
    Args:
        the high level function to create a h5py file,
        mode can be either train, val or test, for the
        train mode, one can set the datasets to be a content
        and each word will be mapping to a index, if the word
        not in the vocabulary, then set skip this or set to
        Unkow regard to the model's setting.
    """
    # we use a entry function to derive the an entry and later
    # repetively use this function to creat the whole training data sets
    def make_entry(article, title):
        article_ids = word2id(article)
        input_ids, output_ids = decoder_inputs(title, vocab)
        return list(article_ids, input_ids, output_ids)
    
    train_data = []
    for i in range(len(data)):
        train_data.append(make_entry(data['content'][i], data['title'][i]))
    with h5py.File('train_data.hdf5', 'w') as f:
        f.create_dataset(name='train_data', data=train_data)
    print("the training data has been successfully created!")

def word2id(vocab, sentence):
    """[using mapping table vocab to map the words to ids]
    
    Parameters
    ----------
    vocab : [conatin word2id and it2word]
    sentence : [a list of strings]
    
    Returns
    -------
    [list]
        [the same length as sentences]
    """

    ids = []
    word2id_ = vocab['word_2_id']
    for word in sentence:
        if word not in word2id_:
            ids.append(word2id_['UNK'])
        else:
            ids.append(word2id_[word])
    return ids


def id2word(vocab, ids):
    sentences = []
    id2word_ = vocab['id_2_word']
    for id in ids:
        if id not in id2word_:
            sentences.append(id2word_[len(id2word_)])
        else:
            sentences.append(id2word_[id])
    return sentences


def encoder_inputs(sentences, vocab):
    """[encoder the sentences of a paragraph to enable them to
    won the ids as inputs.]

    Arguments:
        sentences {list of strings} -- [each item in the string will be a
        word or UNK token.]
        vocab {a dictionary contain word2id and id2word} -- [description]
    """
    word2id_ = vocab['word2id']
    ids = []
    oov = []
    for word in sentences:
        try:
           ids.append(word2id_[word])
        except KeyError as er:
            if config.oov:
                ids.append(word2id_['UNK'])
                oov.append(word)
            else:
                pass
    if config.oov:
        return ids, oov
    else:
        return ids


def decoder_inputs(title, vocab):
    """[decoder_inputs will convert the tile to the inputs of
    the encoder and the output of the encoder so that we will
    have the paired sentences in the decodr]
    
    Parameters
    ----------
    title : [a list of string, each will own several ]
        [description]
    
    """
    title_id = word2id(title, vocab)
    word_2_id = vocab['word_2_id']
    input_ids = word_2_id['<SOS'] + title_id
    output_ids = title_id + word_2_id['<EOS>']
    return input_ids, output_ids
