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
        packed_x = pack_padded_sequence(X, seq_lens, batch_first=True)
        outputs, hidden = self.rnn(packed_x)
        outputs = pad_packed_sequence(outputs, seq_lens, batch_first=True)
        return outputs, hidden
