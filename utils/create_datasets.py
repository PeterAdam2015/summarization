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

def word_2_id(vocab, word):
    word_2_id_ = vocab['word2id']
    if word not in word_2_id_:
        return word_2_id_['UNK']
    return word_2_id_[word]

def id_2_word(vocab, word_id):
    id_2_word_ = vocab['id2word']
    if word_id not in id_2_word_:
        raise ValueError("Id not found in vocab: %d" % word_id)
    return id_2_word_[word_id]

def article_2_ids(article_words, vocab):
    """
    Notice this function only be implemented when using pointer-generator,
    not implemented in navie mode. togather with the abstract_2_ids().
    """
    ids = []
    oovs = []
    unk_id = word_2_id(vocab, 'UNK')
    for w in article_words:
        i = word_2_id(vocab, w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
                oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(len(vocab['word2id'])-1 + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_2_ids(abstrcat_words, vocab, article_oovs):
    """
    Notice this function only be implemented when using pointer-generator,
    not implemented in navie mode. togather with the article_2_ids().
    """
    ids = []
    unk_id = word_2_id(vocab, 'UNK')
    for w in abstract_words:
        i = word_2_id(vocab, w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx =len(vocab['word2id'])-1 + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def abstract2sents(abstract):
    """
    translate the abstract ids to a list of sentences, may be not 
    useful in this match setting.
    """
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index('SOS', cur)
            end_p = abstract.index('EOS', start_p + 1)
            cur = end_p + len('EOS')
            sents.append(abstract[start_p+len('SOS'):end_p])
        except ValueError as e: # no more sentences
            return sents


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
        start_decoding = word_2_id(vocab, 'SOS')
        stop_decoding = word_2_id(vocab, 'EOS')
        # Process the article, the UNK tokens will be replaced as the
        # mapping runs, so we need to keep the all article_words here.
        # the article may not be the totally the same length here, by default
        # we will not use the oov as inforamtion, so we just will have UNK
        # for all unknown words.
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        
        self.enc_len = len(article_words) # store the length after truncation but before padding
        self.enc_input = [word_2_id(vocab, w) for w in article_words]
        # enc_input will contain UNK token mapping
        # Process the abstract, the abstract conists by a lot of sentences.
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [word_2_id(vocab, w) for w in abstract_words]

        # Get the d, ecoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
        
        # the most useful information is the dec_len, dec_input, enc_input, enc_len and the target.
        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by
            # their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = article_2_ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented
            # by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract_2_ids(abstract_words, vocab, self.article_oovs)

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

def process_entry(entry):
    """[process a entry to return the features of the entry]
    
    Parameters
    ----------
    entry : [Entry instance]
    """
    assert isinstance(entry, Entry), "the object must be a Entry instance!"
    # question, for torch, shoud we must specify the length of the inputs ? not really
    # we can always know the shape of the inputs by some operations.
    entry.pad_decoder_inp_targ(config.max_dec_steps, 0)  # set padding id always be 0
    entry.pad_decoder_inp_targ(config.max_dec_steps, 0)
    return (entry.enc_input, entry.dec_input, entry.target)



def save_features(csv_file, vocab, mode):
    """
    Parameters
    ----------
    csv_file : [path name to contain a csv file]
        [the csv file contian either the total train data stored as a csv 
        file with id's head look like the following:
            id      content     title
            1       ....        ....
        for trian mode or the following:
            id      content     title
            1       ....        NULL
        for test mode.
        ]
    vocab : [the vocab]

    mode : [string, either train or test]
        [for evalutaiton mode, or the dev mode, we treat them as 
        train mode too.]
    
    svae the contexts and titles to the standard h5py files
    
    """
    assert mode in ('train', 'test'), "the given mode is ivalid, using only train  or test!"
    data = pd.read_csv(csv_file, encoding = 'utf-8')
    content = list(data['content'])
    if mode == 'train':
        title = list(data['title'])
    else:
        # using a word for dummy titles.
        title = [['title'] for i in range(len(content))]
    list_data = list(zip(content, title))
    features = []
    example_lists = [Entry(entry[0], entry[1], vocab) for entry in list_data]
    for example in example_lists:
        # to check the features, you just need some test on the jupyter notebook
        features.append(process_entry(example))
    return features
    # with h5py.File('features', 'w') as f:
    #     f.create_dataset()
        


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