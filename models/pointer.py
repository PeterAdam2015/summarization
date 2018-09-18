"""
using torch to implement the pointer genreator and it's 
baseline model, here we need just to use the base line model.

"""
import torch
import torch.nn as nn
import utils.config as config


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
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, self.hidden_dim)

    def forward(self, x, return_sequence):
        """forward implementation of the encoder
        
        Parameters
        ----------
        x : [tensor]
            [with the shape been batch_size, max_enc_steps, embedding_dim]
        
        """

    def initialzie(self, data=None):
        """to initialzie all the models in the Encoder, here mainly use LSTM 
           and also use the Embedding matrix, we will initialize all this two
           sub models using either pretrained matrix if the data is given or
           use a random unifom intializer.
        
        Parameters
        ----------
        data : [type]
            [description]
        
        """

        assert self.embedding.weight.data.copy_(torch.from_numpy(data))