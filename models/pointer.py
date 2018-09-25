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
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(config.NUM_WORDS+1, self.embedding_dim, padding_idx=0)
        init_wt_unif(self.embedding)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        init_lstm_wt(self.lstm)

    def forward(self, x):
        """forward implementation of the encoder
        
        Parameters
        ----------
        x : [tensor]
            [with the shape been batch_size, max_enc_steps, embedding_dim]
        if we don't use the variant length of the input sequences.
        """
        embeded_x = self.embedding(x)
        output, hidden = self.lstm(packed_x)
        h = output.contiguous()
        max_h = h.max(dim=1) # the maximum outputs of each time step
        return h, hidden, max_h
    
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(2*config.hidden_dim, config.hidden_dim)
        init_linear_wt(self.reduce_c)
    
    def forward(self, hidden):
        """
        by default, the bidirectional has been applied, so the output tensor of the
        LSTM should be bi*batch_szie*output_hidden_dim, this function will implement the
        non-linearty and also the linear mapping from bidirectional hidden dimension to
        sigal hidden dimension.
        
        Parameters
        ----------
        hidden : [type]
            [a tuple contain both final hidden state and cell state.]
        
        Returns
        -------
        [type]
            reduced hidden state after mapping and non-linearty.
        """
    
        h_, c_ = hidden
        hidden_reduced_h = F.relu(self.reduce_h(h_.view(-1, config.hidden_dim*2)))
        hidden_reduced_c = F.relu(self.reduce_c(c_.view(-1, config.hidden_dim*2)))
        # using unsqueeze to add one additional dimension here.
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
    
    def forward(self, s_t_hat, h, enc_padding_mask, coverage):

        b, t_k, n = list(h.size())
        h = h.view(-1, n)
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_fea_expanded = dec_fea_expanded.view(-1, n)

        att_features = encoder_feature + dec_fea_expanded
        e = F.tanh(att_features)
        scores = self.v(e)
        scores = scores.view(-1, n)

        attn_dist_ = F.softmax(scored, dim=1) * enc_padding_mask
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        # attn_dist is the weight distribution
        attn_dist = attn_dist.unsqueeze(1)
        h = h.view(-1, t_k, n)
        # implement the batch matrix multiply
        c_t = torch.bmm(attn_dist, h)
        c_t = c_t.view(-1, config.hidden_dim * 2)
        
        attn_dist = attn_dist.view(-1, t_k)

        return c_t, attn_dist


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()

        self.embedding = nn.Embedding(config.NUM_WORDS + 2, config.embedding_dim)
        init_wt_normal(self.embedding.weight)

        self.x_content = nn.Linear(config.hidden_dim * 2 + config.embedding_dim, config.embedding_dim)
        
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.NUM_WORDS)
        init_linear_wt(self.out2)

    def forward(self) 