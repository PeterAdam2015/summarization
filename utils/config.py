import os
"""
some configuration of the entire model
"""


vocab_path = '../data/Vocab.pkl'
train_path = '../data/features-600-40_v2.hdf5'
batch_size = 300
epoches = 10
print_every = 10
lr = 1e-4
oov = False

max_enc_steps = 600
max_dec_steps = 40
pointer_gen = False

UNKNOW = 'UNK'
EOS = 'EOS'
SOS = 'SOS'
PAD = 'PAD'
NUM_WORDS = 50000

embedding_dim = 100
hidden_dim = 50
# for the initialzier setting here:
rand_unif_init_rang = 2
trunc_norm_init_std = 1
