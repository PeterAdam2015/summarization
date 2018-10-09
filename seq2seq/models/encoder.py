import torch
import torch.nn as nn
import torch.nn.functional as F

from .EncoderRNN import EncoderRNN

class BaseLine_Encoder(EncoderRNN):
    def __init__(self, BaseLine_Encoder):
    
