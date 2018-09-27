"""
baseline model: contain only encoder and decoder, very simple
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(config.NUM_WORDS + 2, config.embedding_dim)
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim, bidirectional=True)
        self.reduce_ = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        
    def forward(self, X, seq_lens):
        """return the final satets and also the outputs of each timesteps, for the later usage of
        computing the Attentaion matrix foe each time step input of the Decoder.
        
        Parameters
        ----------
        X : [Torch tensor with batch*MAX_STEP]
            
        seq_lens : [descend order of the real length of the data]
        """
        X = self.embed(X)
        batch_size = X.size()[0]
        packed_x = pack_padded_sequence(X, seq_lens, batch_first=True)
        outputs, hidden = self.rnn(packed_x)
        outputs, seq_lens = pad_packed_sequence(outputs, batch_first=True)
        # outputs is a bathc*max_enc_steps*(2*hidden_dim), but for the hidden
        # must give then to batch first format, so we need to implement this
        # with the following code.
        hidden_c, hidden_s = hidden
        hidden_c = self.reduce_(hidden_c.permute([1, 0, 2]).contiguous().view(batch_size, -1))
        hidden_s = self.reduce_(hidden_s.permute([1, 0, 2]).contiguous().view(batch_size, -1))
        
        hidden_c.unsqueeze_(1)
        hidden_s.unsqueeze_(1)
        # to make the hidden to be batch first and later, do the loop over time
        hidden = (hidden_c.permute([1, 0, 2]), hidden_s.permute([1, 0, 2]))
        return outputs, hidden


class Decoder(nn.Module):
    # TODO: add attention mechanism
    def __init__(self):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(config.NUM_WORDS+4, config.embedding_dim)
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.logits = nn.Linear(config.hidden_dim, config.NUM_WORDS+4, bias=False)
        
        
        
    def forward(self, X, hidden):
        """
        using the encoder's hidden state to initiaize the decoder
        input and also use the teaching force in the training mode. here
        TODO: how we tell the real difference between the train, teaching force and
        the evalutaion?
        """
        X = self.embed(X)
        # first we need to transoform the X to be the seq_len first
        X = X.unsqueeze_(1) # just one time step, the X now shoud be batch_size*1*embeeding_dim
        outputs, hidden = self.rnn(X, hidden)
        outputs = F.log_softmax(self.logits(outputs.squeeze_(1)), dim=1)
        return outputs, hidden


# combine all the model togather:
class BaseModel(object):
    def __init__(self, model_file_path=None, is_eval=False):
        self.encoder = Encoder()
        self.decoder = Decoder()

        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
