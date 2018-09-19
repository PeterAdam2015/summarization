import os
"""
some configuration of the entire model
"""

oov = False

max_enc_steps = 600
max_dec_steps = 40
pointer_gen = False

UNKNOW = 'UNK'
EOS = 'EOS'
SOS = 'SOS'
PAD = 'PAD'
NUM_WORDS = 50000

# for the initialzier setting here:
rand_unif_init_rang = 2
trunc_norm_init_std = 1