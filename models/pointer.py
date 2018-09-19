"""
using torch to implement the pointer genreator and it's 
baseline model, here we need just to use the base line model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config

# define some initialzier for the model

def init_lstm_wt(lstm):
    for names in lstm.all_weights:  # all_weights is a RNN attributes, in which stores the names of all weights tensor
        # for pytorch, the lstm.all_weights, each item belong to one layer and one direction.
        for name in names:
            if name.startwith('weight_'):
                wt = getattr(lstm, name)  # self.name = parameter for this name
                wt.data.uniform_(-config.rand_unif_init_rang, config.rand_unif_init_rang)
            elif name.startwith('bias_'):
                # all the weights and bias belong to one total bias
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """initiaize the linear layer
    
    Parameters
    ----------
    linear : [linear, nn.Module.Linear instance]
        [Initalize the linear layer]
    
    """
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(weight):
    weight.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(weight):
    weight.data.uniform_(-config.rand_unif_init_rang, config.rand_unif_init_rang)




class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        """create an Encoder, the contents id will be
        feed into the Encoder and later, use LSTM to extratct all
        the time step hidden states, for 
        
        Parameters
        ----------
        nn : [type]
            [description]
        input_size : [type]
            [description]
        hidden_dim : [type]
            [description]
        
        """

        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(config.NUM_WORDS+1, self.embedding_dim, padding_idx=0)
        init_wt_unif(self.embedding)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        init_lstm_wt(self.lstm)

    def forward(self, x, seq_lens):
        """forward implementation of the encoder
        
        Parameters
        ----------
        x : [tensor]
            [with the shape been batch_size, max_enc_steps, embedding_dim]
        seq_lens: [tensor]
            [with the length of the x for no-pad ids, the order for the seq_lens should be
            descending order.]
        
        """
        embeded_x = self.embedding(x)
        packed_x = pack_padded_sequence(embeded_x, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed_x)
        h, _ = pad_packed_sequence(output, seq_lens, batch_first=True)
        