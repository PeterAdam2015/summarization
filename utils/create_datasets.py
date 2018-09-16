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


def word2id(word, vocab):
    word_2_id = vocab['word2id']
    if word not in word_to_id:
        return word_2_id['UNK']
    return word_to_id[word]

def id2word(word_id, vocab):
    id_2_word = vocab['id2word']
    if word_id not in id_to_word:
        raise ValueError('Id not found in vocab: %d' % word_id)
    return id_2_word[word_id]


def article2ids(article_words, vocab):
    """
    article2ids will not only return the ids of the given  sentences, but also
    keep the oovs in the sentences instead of filtering them out.
    """
    ids = []
    oovs = []
    word_2_id = vocab['word2id']
    unk_id = word_2_id['UNK']
    for w in article_words:
    i = word2id(w, vocab)
    if i == unk_id: # If w is OOV
        # the oov is a unique list of out of vocabulary here.
        if w not in oovs: # Add to list of OOVs
        oovs.append(w)
        oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
        ids.append(len(word_2_id) + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
        ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    word_2_id = vocab['word2id']
    unk_id = word_2_id['UNK']
    for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
        if w in article_oovs: # If w is an in-article OOV
            vocab_idx = len(word_2_id) + article_oovs.index(w) # Map to its temporary article OOV number
            ids.append(vocab_idx)
        else: # If w is an out-of-article OOV
            ids.append(unk_id) # Map to the UNK token id
    else:
        ids.append(i)
    return ids

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


class Entry(object):
    """
    create a entry of a pair of content and a title, the intialize have assume that we
    have used the UNK to replace the unkonw token of the vocabulary in the articles and
    also the abstract_sentences. Also, the abstract_sentences is a list of sentences here.
    """

    def __init__(self, article, abstract_sentences, vocab):
        """
        Args:
            article: sring, contain the content an aritcle
            abstrcat_sentences: a list of sentences
            vocab: a vocab instance 


        """


        # Get ids of special tokens
        start_decoding = '<EOS>'
        stop_decoding = '<SOS>'
        # Process the article, the UNK tokens will be replaced as the
        # mapping runs, so we need to keep the all article_words here.
        # the article may not be the totally the same length here, by default
        # we will not use the oov as inforamtion, so we just will have UNK
        # for all unknown words.
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        
        self.enc_len = len(article_words) # store the length after truncation but before padding
        self.enc_input = [word_2_id(w) for w in article_words]
        # enc_input will contain UNK token mapping
        # Process the abstract, the abstract conists by a lot of sentences.
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
        
        # the most useful information is the dec_len, dec_input, enc_input, enc_len and the target.
        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by
            # their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented
            # by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        # pad the decoder input, if meets, using pad_id 
        # to pad the remain value in the target
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)


class SumDatasets(Dataset):
    """
    for each batch we will ensure that the length of the decoder and the length
    of the encoder is the same. this can be processed in the Entry
    """
    
    def __init__(self):
        super(SumDatasets, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self):
        pass