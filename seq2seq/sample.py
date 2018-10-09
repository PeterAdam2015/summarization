import numpy as np
import pandas as pd
import  torch
import os
import logging
import torchtext

from trainer import SupervisedTrainer
from models import EncoderRNN, DecoderRNN, Seq2seq
from dataset import SourceField, TargetField
from optim import Optimizer
from loss import Perplexity
from evaluator import Predictor
from nltk.tokenize import RegexpTokenizer
from torchtext.data import TabularDataset
from  util.checkpoint import Checkpoint


data_dir = '../data/'
file_name = 'test_data.csv'

# don't need to use the pandas to show the tabular information here.

# personal defined tokenizer
def tokenizer(example):
    example = example.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(example)
    filtered_words = [w for w in tokens]
    return filtered_words

max_encoder_len = 1000
min_decoder_len = 2

content, title = SourceField(tokenize=tokenizer),TargetField(tokenize=tokenizer)

def len_filter(example):
    return len(example.content) <= max_encoder_len and len(example.title) >=min_decoder_len

datafiled = [('id', None), ('content', content), ('title', title)]
train = TabularDataset(path=os.path.join(data_dir, file_name), 
                     format='csv', fields=datafiled, skip_header=True,
                     filter_pred=len_filter)

content.build_vocab(train, max_size=50000)
title.build_vocab(train, max_size=50000)

weight = torch.ones(len(title.vocab))
pad = title.vocab.stoi[title.pad_token]
loss = Perplexity(weight, pad)
loss.cuda()
seq2seq = None
optimizer = None
hidden_size = 128
bidirectional = True

encoder = EncoderRNN(len(content.vocab), max_encoder_len, hidden_size, bidirectional=bidirectional, variable_lengths=True)
decoder = DecoderRNN(len(title.vocab), 20, hidden_size*2 if bidirectional else hidden_size, dropout_p=0.2, use_attention=True, 
                     bidirectional=bidirectional, eos_id = title.eos_id, sos_id = title.sos_id)

seq2seq =Seq2seq(encoder, decoder)
seq2seq.cuda()
for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)

t = SupervisedTrainer(loss=loss, batch_size=60, checkpoint_every=1e3, print_every=100, expt_dir='../data', device=0)
t.train(seq2seq, train, num_epochs=6, dev_data=None, optimizer=optimizer, teacher_forcing_ratio=0.8)


input_vocab = content.vocab
output_vocab = title.vocab


predictor = Predictor(seq2seq, input_vocab, output_vocab)
tese_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'), encoding =
                        'utf-8')
test_contents = list(test_data.content)[:20]
# use the same tokenizer with the training data here.
test_contents =  [tokenizer(content) for content in test_contents]

for content in test_contents:
    print("the predicted title is:\n")
    print(predictor.predict(content))
