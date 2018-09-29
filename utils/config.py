import os
import torch
"""
some configuration of the entire model
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple
vocab_path = '../data/Vocab.pkl'
csv_path = '../data/remove_duplicated_training_data.csv'
batch_size = 1000
epoches = 500
print_every = 20
lr = 1e-3
oov = False
use_gpu = True
device = torch.device("cuda:0") if torch.cuda.is_available() and use_gpu else torch.device("cpu")


max_enc_steps = 600
max_dec_steps = 10
min_enc_steps = 30
min_dec_steps = 2
pointer_gen = False

train_path = '../data/features-{}-{}-v2.hdf5'.format(max_enc_steps, max_dec_steps)
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
